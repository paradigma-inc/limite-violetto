"""Candidate implementation of Violetto's logits head.

Integrated from the runtime-shape sampler research copy of the finished
logits kernel. Full 128-row tiles fuse GEMM and softcap; other batches use
torch GEMM and Triton softcap. The caller retains the shipped path for T=1
and unsupported inputs.

Numerical contract:

* `logits(hidden, weight)` takes `hidden [T, 1280]` bf16 and the tied table
  `weight [151680, 1280]` bf16, and returns `[T, 151680]` float32, contiguous.
* The GEMM result is rounded once to bf16 before the fp32 softcap
  `23 * sigmoid((raw + 5) / 7.5)`.
* No host synchronization, no data-dependent Python control flow, no caching
  of results or of anything derived from `weight` across calls.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5


@triton.jit
def _softcap(raw, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(raw + offsets, mask=mask, other=0).to(tl.float32)
    shifted = values + 5.0
    scaled = tl.div_rn(shifted, 7.5)
    sigmoid = 1.0 / (1.0 + tl.exp(-scaled))
    result = sigmoid * 23.0
    tl.store(out + offsets, result, mask=mask)


@triton.jit
def _gemm_softcap(
    hidden,
    weight,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr = 0,
):
    if GROUP_M > 0:
        # Consecutive programs visit row tiles sharing a vocabulary tile.
        # This dispatch uses M=1024, with GROUP_M dividing its eight row tiles.
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BN)
        group = pid // (GROUP_M * num_n)
        pid_m = group * GROUP_M + pid % GROUP_M
        pid_n = (pid % (GROUP_M * num_n)) // GROUP_M
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        a = tl.load(
            hidden + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + cols[None, :] * K + k[:, None],
            mask=(cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    # Round the completed sum, never a partial sum, before the fp32 softcap.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    shifted = rounded + 5.0
    scaled = tl.div_rn(shifted, 7.5)
    sigmoid = 1.0 / (1.0 + tl.exp(-scaled))
    result = sigmoid * 23.0
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        result,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _logits(hidden: Tensor, weight: Tensor) -> Tensor:
    if hidden.shape[0] == 16:
        m, k = hidden.shape
        n = weight.shape[0]
        out = torch.empty((m, n), dtype=torch.float32, device=hidden.device)
        _gemm_softcap[(1, triton.cdiv(n, 64))](
            hidden,
            weight,
            out,
            m,
            n,
            k,
            BM=16,
            BN=64,
            BK=64,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )
        return out
    if hidden.shape[0] == 1024:
        m, k = hidden.shape
        n = weight.shape[0]
        out = torch.empty((m, n), dtype=torch.float32, device=hidden.device)
        _gemm_softcap[(triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
            hidden,
            weight,
            out,
            m,
            n,
            k,
            BM=128,
            BN=128,
            BK=64,
            GROUP_M=8,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )
        return out
    # Avoid the wasted GEMM work of a partially populated 128-row tile.
    if hidden.shape[0] >= 128 and hidden.shape[0] % 128 == 0:
        m, k = hidden.shape
        n = weight.shape[0]
        out = torch.empty((m, n), dtype=torch.float32, device=hidden.device)
        _gemm_softcap[(triton.cdiv(m, 128), triton.cdiv(n, 128))](
            hidden,
            weight,
            out,
            m,
            n,
            k,
            BM=128,
            BN=128,
            BK=64,
            num_warps=8,
            num_stages=5 if m == 128 else 3,
            enable_fp_fusion=False,
        )
        return out
    # Materializing this bf16 tensor preserves the mandatory GEMM rounding.
    raw = F.linear(hidden, weight)
    out = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
    n_elements = raw.numel()
    block = 512 if hidden.shape[0] == 320 else 1024
    _softcap[(triton.cdiv(n_elements, block),)](
        raw, out, n_elements, BLOCK=block, num_warps=4, enable_fp_fusion=False
    )
    return out


# M and n_elements remain runtime arguments to avoid batch-drain recompiles.
# These sizes cover every applicable kernel configuration, including both
# softcap blocks and both ordinary 128-row GEMM pipeline depths. T=1 stays
# on the shipped path and has no custom kernel to warm.
_WARM_SIZES = (16, 128, 256, 1024, 320, 7)
_warmed_devices: set[torch.device] = set()


def logits(hidden: Tensor, weight: Tensor) -> Tensor:
    # vLLM calls compute_logits during startup before accepting requests.
    # Cache only compilation readiness, never outputs or weight-derived data.
    if hidden.device not in _warmed_devices:
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Warming logits-head dispatch branches: %s", _WARM_SIZES)
        for rows in _WARM_SIZES:
            _logits(
                torch.zeros(
                    (rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
                ),
                weight,
            )
        _warmed_devices.add(hidden.device)
        logger.info("Logits-head dispatch branch warm-up complete")
    return _logits(hidden, weight)
