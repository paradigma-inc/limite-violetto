"""Violetto attention epilogue: XSA correction and output gate in one launch.

Imported from the `violetto-attn-kernels` incumbent; see `PROVENANCE.md`.
Every `.to(tl.bfloat16)` reproduces a rounding of the shipped bf16 graph.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def _tiled(
    Y,
    V,
    A,
    Alpha,
    W,
    Out,
    YT: tl.constexpr,
    YH: tl.constexpr,
    YD: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    AS: tl.constexpr,
    T,
    BLOCK_T: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    valid = token[:, None] < T
    y = tl.load(
        Y + token[:, None] * YT + head * YH + d[None, :] * YD, valid, other=0
    ).to(tl.float32)
    v = tl.load(
        V + token[:, None] * VT + (head // 5) * VH + d[None, :] * VD, valid, other=0
    ).to(tl.float32)
    norm = tl.sqrt(tl.sum(v * v, 1))
    vn = v / tl.maximum(norm[:, None], 1.0e-4)
    proj = tl.sum(y * vn, 1)
    alpha = libdevice.tanh(tl.load(Alpha + head * AS).to(tl.float32))
    correction = (alpha * proj[:, None] * vn).to(tl.bfloat16).to(tl.float32)
    corrected = (y - correction).to(tl.bfloat16).to(tl.float32)
    a = tl.load(A + token[:, None] * AT + d[None, :] * AD, valid, other=0).to(
        tl.float32
    )
    w = tl.load(W + head * WH + d * WD).to(tl.bfloat16).to(tl.float32)
    gate_logit = tl.sum(a * w[None, :], 1).to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-gate_logit))).to(tl.bfloat16).to(tl.float32)
    gate = (2.0 * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Out + token[:, None] * 1280 + head * 128 + d[None, :],
        corrected * gate[:, None],
        valid,
    )


@tr.jit
def _grouped_unrolled(
    Y,
    V,
    A,
    Alpha,
    W,
    Out,
    YT: tl.constexpr,
    YH: tl.constexpr,
    YD: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    AS: tl.constexpr,
    T,
    BT: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    kv = tl.program_id(1)
    d = tl.arange(0, 128)
    valid = token[:, None] < T
    v = tl.load(V + token[:, None] * VT + kv * VH + d[None, :] * VD, valid, other=0).to(
        tl.float32
    )
    norm = tl.sqrt(tl.sum(v * v, 1))
    vn = v / tl.maximum(norm[:, None], 1.0e-4)
    a = tl.load(A + token[:, None] * AT + d[None, :] * AD, valid, other=0).to(
        tl.float32
    )
    for h in tl.static_range(5):
        head = kv * 5 + h
        y = tl.load(
            Y + token[:, None] * YT + head * YH + d[None, :] * YD, valid, other=0
        ).to(tl.float32)
        proj = tl.sum(y * vn, 1)
        alpha = libdevice.tanh(tl.load(Alpha + head * AS).to(tl.float32))
        correction = (alpha * proj[:, None] * vn).to(tl.bfloat16).to(tl.float32)
        corrected = (y - correction).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + head * WH + d * WD).to(tl.bfloat16).to(tl.float32)
        logit = tl.sum(a * w[None, :], 1).to(tl.bfloat16).to(tl.float32)
        gate = (1.0 / (1.0 + tl.exp(-logit))).to(tl.bfloat16).to(tl.float32)
        gate = (2.0 * gate).to(tl.bfloat16).to(tl.float32)
        tl.store(
            Out + token[:, None] * 1280 + head * 128 + d[None, :],
            corrected * gate[:, None],
            valid,
        )


def epilogue(y, v, attn_in, xsa_alpha, attn_gate):
    """`y [T,10,128]`, post-VE `v [T,2,128]` -> gated, XSA-corrected `[T,1280]`."""
    tokens = y.shape[0]
    out = torch.empty((tokens, 1280), dtype=y.dtype, device=y.device)
    strides = (
        *y.stride(),
        *v.stride(),
        *attn_in.stride(),
        *attn_gate.stride(),
        xsa_alpha.stride(0),
    )
    if tokens < 512:
        bt = 1 if tokens < 16 else 2
        _tiled[(tr.cdiv(tokens, bt), 10)](
            y,
            v,
            attn_in,
            xsa_alpha,
            attn_gate,
            out,
            *strides,
            tokens,
            bt,
            num_warps=1,
            enable_fp_fusion=False,
        )
    elif tokens == 512:
        _tiled[(tr.cdiv(tokens, 8), 10)](
            y,
            v,
            attn_in,
            xsa_alpha,
            attn_gate,
            out,
            *strides,
            tokens,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _grouped_unrolled[(tr.cdiv(tokens, 8), 2)](
            y,
            v,
            attn_in,
            xsa_alpha,
            attn_gate,
            out,
            *strides,
            tokens,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
