# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.28.0 Gumbel sampling that skips blocks top-p masked entirely.

A copy of upstream ``vllm/v1/worker/gpu/sample/gumbel.py``'s sample kernel. A
1024-token block whose logits are all ``-inf`` gets no noise: it emits ``-inf``
at the block's first token, exactly what upstream's finite noise and leftmost
tie-break produce there. Philox noise is counter-based, so every other block
draws the same values as upstream. Only :func:`gumbel_sample` is installed, and
only into the seeded sampler's private namespace (:mod:`limite_vllm.seeded_sampler`);
the upstream module stays untouched as the equivalence oracle.
"""

import hashlib
import logging
import time
from pathlib import Path

import torch
import vllm.v1.worker.gpu.sample.gumbel as _original
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    # All lanes, including padding, are -inf. Upstream adds finite noise
    # and reduces tied -inf values to the leftmost lane (idx=0).
    # No cache write may be skipped; callers with a cache use upstream.
    if tl.sum((logits != float("-inf")).to(tl.int32), axis=0) == 0:
        value = float("-inf")
        idx = 0
    else:
        value, idx = gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            logits_cache_ptr,
            logits_cache_stride,
            logits_cache_col_ptr,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
            USE_FP64=USE_FP64,
            PER_TOKEN_COL=PER_TOKEN_COL,
        )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def _masked_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


_LOG = logging.getLogger(__name__)
_COUNTS = [0, 0]
_LAST_LOG = 0.0
_CHECKING = False


def check_source():
    expected = "1f9ca94ccaa27ac6c7b15bfcd8342a9118e2e624f2b5fa7fd19e5eee8614a823"
    if hashlib.sha256(Path(_original.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError("masked Gumbel patch: upstream source changed")


def gumbel_sample(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    logits_cache=None,
    logits_cache_col=None,
    use_fp64=False,
):
    global _LAST_LOG
    supported = (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[1] > 0
        and logits.stride(1) == 1
        and not apply_temperature
        and logits_cache is None
        and logits_cache_col is None
        and not use_fp64
    )
    if not _CHECKING:
        _COUNTS[0 if supported else 1] += 1
        now = time.monotonic()
        if now - _LAST_LOG >= 30:
            _LOG.warning(
                "Limite Gumbel dispatch counters patched=%d fallback=%d", *_COUNTS
            )
            _LAST_LOG = now
    fn = _masked_sample if supported else _original.gumbel_sample
    return fn(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        logits_cache,
        logits_cache_col,
        use_fp64,
    )


def check_blocks(device):
    """Startup-only: compare intermediate block values/indices and final ids.

    Includes all-masked rows, partial tail blocks, exact ties within and across
    blocks, boundary survivors, greedy and stochastic rows and varied positions.
    The all-masked row must pick token zero; a greedy tie must pick the first
    token, both within a block and across blocks.
    """
    rows, vocab = 7, 3075
    x = torch.full((rows, vocab), -float("inf"), device=device)
    x[1, 1023:1025] = 22.0
    x[2, 2047:2049] = 22.0
    x[3, 3072:3075] = 22.0
    x[4, :] = 1.0
    x[5, 1023:1025] = 22.0
    x[6, 2047:2049] = 22.0
    mapping = torch.arange(rows, device=device, dtype=torch.int32)
    temp = torch.tensor([0.6, 0.0, 0.0, 1.0, 1.0, 0.6, 1.0], device=device)
    seeds = torch.arange(rows, device=device, dtype=torch.int64) * 7919 + 113
    num_blocks = triton.cdiv(vocab, 1024)
    cases = 0
    for offset in (0, 19, 4097):
        pos = torch.arange(rows, device=device, dtype=torch.int64) * 17 + offset
        outputs = []
        for kernel in (_original._gumbel_sample_kernel, _gumbel_sample_kernel):
            ids = torch.empty((rows, num_blocks), device=device, dtype=torch.int64)
            values = torch.empty((rows, num_blocks), device=device)
            kernel[(rows, num_blocks)](
                ids,
                ids.stride(0),
                values,
                values.stride(0),
                None,
                0,
                None,
                x,
                x.stride(0),
                mapping,
                seeds,
                pos,
                temp,
                vocab,
                BLOCK_SIZE=1024,
                APPLY_TEMPERATURE=False,
                USE_FP64=False,
                PER_TOKEN_COL=False,
            )
            outputs.append((ids, values))
        for i in (0, 1):
            if not torch.equal(outputs[0][i], outputs[1][i]):
                raise RuntimeError("masked Gumbel block value/index mismatch")
        ids, values = outputs[1]
        expected = torch.arange(num_blocks, device=device) * 1024
        if (
            not torch.equal(ids[0], expected)
            or not torch.isneginf(values[0]).all().item()
        ):
            raise RuntimeError("all-masked block did not emit -inf / block start")
        ref = _original.gumbel_sample(x, mapping, temp, seeds, pos, False)
        got = gumbel_sample(x, mapping, temp, seeds, pos, False)
        if not torch.equal(ref, got) or not torch.equal(
            got[:3], torch.tensor([0, 1023, 2047], device=device)
        ):
            raise RuntimeError("masked Gumbel final argmax/tie mismatch")
        cases += rows
    _LOG.warning(
        "Limite Gumbel block equivalence PASS cases=%d exact_block_values_indices=True all_masked_and_leftmost_ties=True",
        cases,
    )
