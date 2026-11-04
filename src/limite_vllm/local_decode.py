"""Triton decode for Limite's sliding-window layers on vLLM 0.28.0.

Pure-decode batches of the 36 sliding-window layers (bf16, causal, one query per
request, a 1,025-key window) run :func:`local_decode`, a split-KV paged-decode
kernel that reads only the window from vLLM's paged KV cache. Prefill, mixed
batches, cascade attention, a quantized KV cache and every other input keep
vLLM's FlashAttention path; global layers always do.

The backend subclasses vLLM 0.28.0's ``FlashAttentionImpl`` and reads its metadata
fields, so :func:`require_runtime` refuses any other vLLM build before a layer is
constructed.
"""

from __future__ import annotations

import hashlib
import logging
from importlib.metadata import version
from pathlib import Path

import torch
import triton
import triton.language as tl
import vllm.v1.attention.backends.flash_attn as _flash_attn
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

logger = logging.getLogger(__name__)

SUPPORTED_VLLM_VERSION = "0.28.0"
# SHA-256 of vLLM 0.28.0's vllm/v1/attention/backends/flash_attn.py, which owns
# FlashAttentionImpl.forward (subclassed below) and the metadata it reads. A build
# that keeps the version string but changes that file fails instead of running
# code written against other internals.
_FLASH_ATTN_SOURCE_SHA = (
    "44f4f0b560f0b09575d52a9e7a44f064bb600796079edd548aab2e21b4f31064"
)


@triton.jit
def _decode_kernel(
    Q,
    KV,
    BT,
    SEQ,
    OPART,
    MPART,
    LPART,
    OUT,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    stride_kvb,
    stride_kvt,
    stride_kvh,
    stride_btb,
    stride_ps,
    stride_pb,
    stride_ph,
    stride_ms,
    stride_mb,
    scale,
    W: tl.constexpr,
    BS: tl.constexpr,
    TILE: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    L = tl.load(SEQ + b)
    start = tl.maximum(L - W, 0)
    t0 = (start // TILE) * TILE
    ntiles = tl.cdiv(L - t0, TILE)
    per = tl.cdiv(ntiles, S)
    tb = s * per
    te = tl.minimum(tb + per, ntiles)
    rows = tl.arange(0, GP)
    dd = tl.arange(0, D)
    qmask = rows < G
    qh = h * G + rows
    q = tl.load(
        Q + b * stride_qb + qh[:, None] * stride_qh + dd[None, :],
        mask=qmask[:, None],
        other=0.0,
    )
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    nn = tl.arange(0, TILE)
    for t in range(tb, te):
        tok = t0 + t * TILE + nn
        valid = (tok >= start) & (tok < L)
        page = tl.load(BT + b * stride_btb + tok // BS, mask=valid, other=0)
        base = page.to(tl.int64) * stride_kvb + (tok % BS) * stride_kvt + h * stride_kvh
        k = tl.load(KV + base[:, None] + dd[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(
            KV + base[:, None] + D + dd[None, :], mask=valid[:, None], other=0.0
        )
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(valid[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    if S == 1:
        out = (
            acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        )  # padded rows (seq_len 0) -> 0
        tl.store(
            OUT + b * stride_ob + qh[:, None] * stride_oh + dd[None, :],
            out.to(tl.bfloat16),
            mask=qmask[:, None],
        )
    else:
        tl.store(
            OPART
            + s * stride_ps
            + b * stride_pb
            + qh[:, None] * stride_ph
            + dd[None, :],
            acc,
            mask=qmask[:, None],
        )
        tl.store(MPART + s * stride_ms + b * stride_mb + qh, m_i, mask=qmask)
        tl.store(LPART + s * stride_ms + b * stride_mb + qh, l_i, mask=qmask)


@triton.jit
def _combine_kernel(
    OPART,
    MPART,
    LPART,
    OUT,
    stride_ps,
    stride_pb,
    stride_ph,
    stride_ms,
    stride_mb,
    stride_ob,
    stride_oh,
    S: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    hq = tl.program_id(1)
    ss = tl.arange(0, S)
    dd = tl.arange(0, D)
    m = tl.load(MPART + ss * stride_ms + b * stride_mb + hq)
    lsum = tl.load(LPART + ss * stride_ms + b * stride_mb + hq)
    mmax = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - mmax))
    o = tl.load(
        OPART + ss[:, None] * stride_ps + b * stride_pb + hq * stride_ph + dd[None, :]
    )
    den = tl.sum(lsum * w, 0)
    out = tl.sum(o * w[:, None], 0) / tl.where(den == 0.0, 1.0, den)
    tl.store(OUT + b * stride_ob + hq * stride_oh + dd, out.to(tl.bfloat16))


_ANNOUNCED = False


def _splits_for(num_reqs: int) -> int:
    if num_reqs >= 128:
        return 1
    if num_reqs >= 64:
        return 2
    if num_reqs >= 16:
        return 4
    return 8


def local_decode(q, kv_cache, block_table, seq_lens, out, scale, window):
    """q/out: (B, HQ, D) bf16 views; kv_cache: vLLM logical (num_blocks, HK, block_size, 2*D)."""
    global _ANNOUNCED
    num_reqs = seq_lens.shape[0]
    if not _ANNOUNCED:
        # One-time dispatch evidence. vLLM leaves non-vllm loggers at WARNING,
        # so a lower level would not reach the server log.
        logger.warning(
            "Limite local decode first launch: num_reqs=%d kv_cache=%s strides=%s window=%d",
            num_reqs,
            tuple(kv_cache.shape),
            kv_cache.stride(),
            window,
        )
        _ANNOUNCED = True
    hq, d = q.shape[1], q.shape[2]
    hk, block_size = kv_cache.shape[1], kv_cache.shape[2]
    splits = _splits_for(num_reqs)
    tile = 32 if block_size % 32 == 0 or 32 % block_size == 0 else 16
    if splits > 1:
        opart = torch.empty(
            (splits, num_reqs, hq, d), dtype=torch.float32, device=q.device
        )
        mpart = torch.empty(
            (splits, num_reqs, hq), dtype=torch.float32, device=q.device
        )
        lpart = torch.empty_like(mpart)
    else:
        opart = mpart = lpart = out  # unused
    _decode_kernel[(num_reqs, hk, splits)](
        q,
        kv_cache,
        block_table,
        seq_lens,
        opart,
        mpart,
        lpart,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(1),
        block_table.stride(0),
        opart.stride(0) if splits > 1 else 0,
        opart.stride(1) if splits > 1 else 0,
        opart.stride(2) if splits > 1 else 0,
        mpart.stride(0) if splits > 1 else 0,
        mpart.stride(1) if splits > 1 else 0,
        scale,
        W=window,
        BS=block_size,
        TILE=tile,
        G=hq // hk,
        GP=16,
        D=d,
        S=splits,
        num_warps=4,
        num_stages=2,
    )
    if splits > 1:
        _combine_kernel[(num_reqs, hq)](
            opart,
            mpart,
            lpart,
            out,
            opart.stride(0),
            opart.stride(1),
            opart.stride(2),
            mpart.stride(0),
            mpart.stride(1),
            out.stride(0),
            out.stride(1),
            S=splits,
            D=d,
            num_warps=4,
        )


class LimiteLocalDecodeImpl(FlashAttentionImpl):
    """FA3 impl whose sliding-window pure-decode batches use :func:`local_decode`."""

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if (
            attn_metadata is not None
            and output_scale is None
            and output_block_scale is None
            and not attn_metadata.use_cascade
            and attn_metadata.max_query_len == 1
            and self.sliding_window == (1024, 0)
            and attn_metadata.causal is True
            and attn_metadata.rswa_prefix_lens is None
            and attn_metadata.mm_prefix_query_range_tensor is None
            and self.alibi_slopes is None
            and self.dcp_world_size == 1
            and not self.batch_invariant_enabled
            and self.vllm_flash_attn_version == 3
            and query.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and query.ndim == 3
            and tuple(query.shape[1:]) == (10, 128)
            and output.ndim == 3
            and tuple(output.shape[1:]) == (10, 128)
            and kv_cache.ndim == 4
            and kv_cache.shape[1] == 2
            and kv_cache.shape[-1] == 256
            and kv_cache.shape[2] > 0
            and kv_cache.shape[2] % 16 == 0
            and query.stride(-1) == output.stride(-1) == kv_cache.stride(-1) == 1
            and attn_metadata.block_table.stride(-1) == 1
            and attn_metadata.seq_lens.stride(0) == 1
            and 0 < attn_metadata.seq_lens.shape[0] <= query.shape[0]
            and attn_metadata.seq_lens.shape[0] <= output.shape[0]
            and not self.logits_soft_cap
            and self.sinks is None
        ):
            num_reqs = attn_metadata.seq_lens.shape[0]
            local_decode(
                query[:num_reqs],
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                output[:num_reqs],
                self.scale,
                self.sliding_window[0] + 1,
            )
            return output
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class LimiteLocalDecodeBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return LimiteLocalDecodeImpl


def backend_for(is_global: bool):
    """Attention backend override for one Limite layer, or None for vLLM's choice."""
    if is_global:
        return None
    return LimiteLocalDecodeBackend


def require_runtime() -> None:
    """Fail before any layer is built unless vLLM is the verified 0.28.0 build."""
    installed = version("vllm")
    if installed.split("+", 1)[0] != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"limite_vllm's attention backend requires vLLM {SUPPORTED_VLLM_VERSION}, "
            f"got {installed}; install the locked runtime (uv sync --locked)."
        )
    source = Path(_flash_attn.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != _FLASH_ATTN_SOURCE_SHA:
        raise RuntimeError(
            "limite_vllm's attention backend: vLLM's flash_attn.py differs from the "
            f"verified {SUPPORTED_VLLM_VERSION} source"
        )
