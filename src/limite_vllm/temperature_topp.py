# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone top-p from pinned vLLM 0.28.0, with exact temperature fusion.

Top-k code is omitted because the caller requires top-k disabled. All pivot,
reduction and duplicate-cutoff operations retain upstream order. Zero/one
temperatures skip division; p=1 rows still receive temperature scaling.
Auxiliary processing and logprob consumers are excluded by seeded_sampler._eligible.

Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
using Pivot-based Truncation and Selection" By Park et al.
(https://arxiv.org/abs/2602.01518)

"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.sample.ops import topk_topp_triton as _upstream
from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats


@triton.jit
def _scale(logits, temp):
    if temp != 0.0 and temp != 1.0:
        # Same fp32 fdiv as upstream _temperature_kernel. The opaque fp32
        # identity prevents later top-p arithmetic from contracting/reassociating
        # across the rounding boundary previously imposed by the store/load.
        logits = logits / temp
        logits = tl.inline_asm_elementwise(
            "mov.b32 $0, $1;",
            constraints="=f,f",
            args=[logits],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
    return logits


@triton.jit
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    TEMPERATURE,
    MAPPING,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE
        req = tl.load(MAPPING + row_id)
        temp = tl.load(TEMPERATURE + req).to(tl.float32)

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                logits_blk0 = _scale(logits_blk0.to(tl.float32), temp)
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    logits_blk = _scale(logits_blk.to(tl.float32), temp)
                    # Later passes load these rounded values without rescaling.
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

            elif temp != 0.0 and temp != 1.0:
                # A top-p=1 row still needs upstream temperature processing.
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    logits_blk = _scale(logits_blk.to(tl.float32), temp)
                    tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def apply(logits, p, temperature, mapping):
    rows, vocab = logits.shape
    num_sm = num_compute_units(logits.device.index)
    programs = min(num_sm, rows)
    key = (logits.device, logits.dtype, vocab)
    buffer = _upstream._TRITON_BUFFER_CACHE.get(key)
    if buffer is None or buffer.shape[0] < programs:
        buffer = logits.new_empty((min(next_power_of_2(programs), num_sm), vocab))
        _upstream._TRITON_BUFFER_CACHE[key] = buffer
    # No lazy host-to-device table initialization on the serving path.
    normal, percentile = _upstream._TRITON_TABLE_CACHE[logits.device]
    _topk_topp_kernel[(programs,)](
        logits,
        logits.stride(0),
        buffer,
        percentile,
        normal,
        logits,
        p,
        temperature,
        mapping,
        BATCH_SIZE=rows,
        VOCAB_SIZE=vocab,
        MASK_VALUE=-float("inf"),
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=False,
        TOPP_ENABLED=True,
        num_warps=8,
    )
    return logits


def eligible(s, logits, idx):
    import numpy as np

    return (
        logits.shape[0] >= 8
        and np.any(s.sampling_states.top_p.np[idx] < 1.0)
        and logits.device in _upstream._TRITON_TABLE_CACHE
    )


def check_source():
    import hashlib
    from pathlib import Path

    expected = "255619f699898ddaaac959d9bef4494294c76d730abb8e90549afac83a9bcbe9"
    if hashlib.sha256(Path(_upstream.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError("temperature top-p patch: upstream source changed")


def check_exact(device, vocab):
    import logging
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature

    cases = 0
    # Explicit cutoff ties, flat and peaked inputs; mixed zero/one/0.6
    # temperatures and mixed p=0.95/1.0. Original matrix covers all row counts.
    for rows in (8, 255, 256, 257, 320, 512):
        idx = torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)
        seq = torch.arange(rows, device=device)
        temps = torch.tensor([0.6, 1.0, 0.0], device=device).repeat((rows + 2) // 3)[
            :rows
        ]
        for pattern in ("flat", "peaked", "cutoff_ties", "masked"):
            col = torch.arange(vocab, device=device, dtype=torch.float32)
            if pattern == "flat":
                x = (0.1 + (col % 101) * 0.0001).expand(rows, -1).clone()
            elif pattern == "cutoff_ties":
                x = (1.0 + (col % 3)).expand(rows, -1).clone()
            else:
                x = torch.full((rows, vocab), 0.1, device=device)
                x[seq, (seq * 997 + 1023) % vocab] = 22.9
            if pattern == "masked":
                x[:, ::2] = -float("inf")
                x[0, :] = -float("inf")
            for mixed in ("none", "mixed", "all"):
                p = torch.full((rows,), 0.95, device=device)
                if mixed == "mixed":
                    p[::3] = 1.0
                elif mixed == "all":
                    p.fill_(1.0)
                ref, got = x.clone(), x.clone()
                apply_temperature(ref, idx, temps)
                _upstream.apply_top_k_top_p_triton(ref, None, p)
                apply(got, p, temps, idx)
                if not torch.equal(torch.isneginf(ref), torch.isneginf(got)):
                    raise RuntimeError(
                        f"temperature top-p mask mismatch: {rows, pattern, mixed}"
                    )
                if not torch.equal(ref, got):
                    raise RuntimeError(
                        f"temperature top-p scaled logits mismatch: {rows, pattern, mixed}"
                    )
                cases += 1
    logging.getLogger(__name__).warning(
        "Limite temperature top-p equivalence PASS cases=%d exact_masks_and_logits=True",
        cases,
    )
