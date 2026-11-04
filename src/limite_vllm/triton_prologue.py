"""Violetto attention prologue: value embeddings, QK RMS norm, partial rotary.

Imported from the `violetto-attn-kernels` incumbent; see `PROVENANCE.md`.
Every `.to(tl.bfloat16)` reproduces a rounding of the shipped bf16 graph.
Q and K are written as one `[T, 12, 128]` buffer: heads 0-9 are Q, 10-11 K.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def _normalize_rotate(
    QKV,
    Positions,
    Freq,
    Out,
    QT: tl.constexpr,
    QD: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    ROPE: tl.constexpr,
    HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    d = tl.arange(0, 128)
    x = tl.load(QKV + token * QT + (head[:, None] * 128 + d[None, :]) * QD).to(
        tl.float32
    )
    inv = tl.rsqrt(tl.sum(x * x, 1) / 128.0 + 1.1920928955078125e-7)
    norm = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        position = tl.load(Positions + token * PS).to(tl.float32)
        theta = position * tl.load(Freq + d * FS)
        cos = libdevice.cos(theta).to(tl.bfloat16).to(tl.float32)
        sin = libdevice.sin(theta).to(tl.bfloat16).to(tl.float32)
        sin = tl.where(d % 2 == 1, -sin, sin)
        partner = tl.gather(norm, tl.broadcast_to((d ^ 1)[None, :], (HEADS, 128)), 1)
        left = (cos[None, :] * norm).to(tl.bfloat16).to(tl.float32)
        right = (sin[None, :] * partner).to(tl.bfloat16).to(tl.float32)
        norm = left + right
    tl.store(Out + token * 1536 + head[:, None] * 128 + d[None, :], norm)


@tr.jit
def _tiled_normalize_rotate(
    QKV,
    Positions,
    Freq,
    Out,
    QT: tl.constexpr,
    QD: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    ROPE: tl.constexpr,
    T,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 128)
    offsets = (
        token[:, None, None] * QT + (head[None, :, None] * 128 + d[None, None, :]) * QD
    )
    x = tl.load(QKV + offsets, token[:, None, None] < T, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 2) / 128.0 + 1.1920928955078125e-7)
    norm = (x * inv[:, :, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        position = tl.load(Positions + token * PS, token < T, other=0).to(tl.float32)
        theta = position[:, None] * tl.load(Freq + d * FS)[None, :]
        cos = libdevice.cos(theta).to(tl.bfloat16).to(tl.float32)
        sin = libdevice.sin(theta).to(tl.bfloat16).to(tl.float32)
        sin = tl.where(d[None, :] % 2 == 1, -sin, sin)
        partner = tl.gather(
            norm, tl.broadcast_to((d ^ 1)[None, None, :], (BT, BH, 128)), 2
        )
        left = (cos[:, None, :] * norm).to(tl.bfloat16).to(tl.float32)
        right = (sin[:, None, :] * partner).to(tl.bfloat16).to(tl.float32)
        norm = left + right
    tl.store(
        Out
        + token[:, None, None] * 1536
        + head[None, :, None] * 128
        + d[None, None, :],
        norm,
        token[:, None, None] < T,
    )


def prologue(qkv, positions, freq, *, uses_rope):
    """Normalized (and rotated) Q/K for a layer without value embeddings."""
    tokens = qkv.shape[0]
    qk = torch.empty((tokens, 12, 128), dtype=qkv.dtype, device=qkv.device)
    if tokens >= 512:
        _tiled_normalize_rotate[(tr.cdiv(tokens, 4), 3)](
            qkv,
            positions,
            freq,
            qk,
            *qkv.stride(),
            positions.stride(0),
            freq.stride(0),
            uses_rope,
            tokens,
            4,
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif tokens >= 16:
        _tiled_normalize_rotate[(tr.cdiv(tokens, 2), 12)](
            qkv,
            positions,
            freq,
            qk,
            *qkv.stride(),
            positions.stride(0),
            freq.stride(0),
            uses_rope,
            tokens,
            2,
            1,
            num_warps=2,
            enable_fp_fusion=False,
        )
    else:
        _normalize_rotate[(tokens, 12)](
            qkv,
            positions,
            freq,
            qk,
            *qkv.stride(),
            positions.stride(0),
            freq.stride(0),
            uses_rope,
            1,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return qk


@tr.jit
def _value_embedding(
    QKV,
    A,
    IDs,
    Table,
    W,
    Out,
    QT: tl.constexpr,
    QD: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    IDS: tl.constexpr,
    TR: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    HEAD_OFFSET: tl.constexpr = 0,
):
    token = tl.program_id(0)
    head = tl.program_id(1) - HEAD_OFFSET
    c = tl.arange(0, 16)
    a = tl.load(A + token * AT + c * AD, c < 12, other=0).to(tl.float32)
    w = tl.load(W + head * WH + c * WD, c < 12, other=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(a * w, 0).to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-logit))).to(tl.bfloat16).to(tl.float32)
    gate = (2.0 * gate).to(tl.bfloat16).to(tl.float32)
    row = tl.load(IDs + token * IDS)
    d = tl.arange(0, 128)
    ve = tl.load(Table + row * TR + (head * 128 + d) * TD).to(tl.float32)
    v = tl.load(QKV + token * QT + (1536 + head * 128 + d) * QD).to(tl.float32)
    update = (gate * ve).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token * 256 + head * 128 + d, v + update)


@tr.jit
def _combined_prologue(
    QKV,
    Positions,
    Freq,
    QK,
    A,
    IDs,
    Table,
    W,
    V,
    QT: tl.constexpr,
    QD: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    IDS: tl.constexpr,
    TR: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    ROPE: tl.constexpr,
):
    # Each program owns exactly one Q/K head or one value head.
    # These disjoint stores require no inter-program synchronization.
    if tl.program_id(1) < 12:
        _normalize_rotate(QKV, Positions, Freq, QK, QT, QD, PS, FS, ROPE, 1)
    else:
        _value_embedding(
            QKV, A, IDs, Table, W, V, QT, QD, AT, AD, IDS, TR, TD, WH, WD, 12
        )


@tr.jit
def _tiled_value_embedding(
    QKV,
    A,
    IDs,
    Table,
    W,
    Out,
    QT: tl.constexpr,
    QD: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    IDS: tl.constexpr,
    TR: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    T,
    BT: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.arange(0, 2)
    c = tl.arange(0, 16)
    a = tl.load(
        A + token[:, None] * AT + c[None, :] * AD,
        (token[:, None] < T) & (c[None, :] < 12),
        other=0,
    ).to(tl.float32)
    w = (
        tl.load(W + head[:, None] * WH + c[None, :] * WD, c[None, :] < 12, other=0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    logit = tl.sum(a[:, None, :] * w[None, :, :], 2).to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-logit))).to(tl.bfloat16).to(tl.float32)
    gate = (2.0 * gate).to(tl.bfloat16).to(tl.float32)
    row = tl.load(IDs + token * IDS, token < T, other=0)
    d = tl.arange(0, 128)
    channel = head[:, None] * 128 + d[None, :]
    ve = tl.load(
        Table + row[:, None, None] * TR + channel[None, :, :] * TD,
        token[:, None, None] < T,
        other=0,
    ).to(tl.float32)
    v = tl.load(
        QKV + token[:, None, None] * QT + (1536 + channel[None, :, :]) * QD,
        token[:, None, None] < T,
        other=0,
    ).to(tl.float32)
    update = (gate[:, :, None] * ve).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Out + token[:, None, None] * 256 + channel[None, :, :],
        v + update,
        token[:, None, None] < T,
    )


@tr.jit
def _large_combined_prologue(
    QKV,
    Positions,
    Freq,
    QK,
    A,
    IDs,
    Table,
    W,
    V,
    QT: tl.constexpr,
    QD: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    IDS: tl.constexpr,
    TR: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    ROPE: tl.constexpr,
    T,
):
    if tl.program_id(1) < 3:
        _tiled_normalize_rotate(QKV, Positions, Freq, QK, QT, QD, PS, FS, ROPE, T, 4, 4)
    elif tl.program_id(0) < tl.cdiv(T, 8):
        # One eight-token V tile per program; no Q/K program writes V.
        _tiled_value_embedding(
            QKV, A, IDs, Table, W, V, QT, QD, AT, AD, IDS, TR, TD, WH, WD, T, 8
        )


@tr.jit
def _small_tiled_combined(
    QKV,
    Positions,
    Freq,
    QK,
    A,
    IDs,
    Table,
    W,
    V,
    QT: tl.constexpr,
    QD: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    AT: tl.constexpr,
    AD: tl.constexpr,
    IDS: tl.constexpr,
    TR: tl.constexpr,
    TD: tl.constexpr,
    WH: tl.constexpr,
    WD: tl.constexpr,
    ROPE: tl.constexpr,
    T,
):
    if tl.program_id(1) < 12:
        _tiled_normalize_rotate(QKV, Positions, Freq, QK, QT, QD, PS, FS, ROPE, T, 2, 1)
    else:
        # Only program 12 writes both V heads for this two-token tile.
        _tiled_value_embedding(
            QKV, A, IDs, Table, W, V, QT, QD, AT, AD, IDS, TR, TD, WH, WD, T, 2
        )


def prologue_ve(
    qkv, attn_in, positions, input_ids, ve_table, ve_gate, freq, *, uses_rope
):
    """Normalized (and rotated) Q/K plus the value-embedded V, in one launch."""
    if qkv.shape[0] < 512:
        qk = torch.empty((qkv.shape[0], 12, 128), dtype=qkv.dtype, device=qkv.device)
        v = torch.empty((qkv.shape[0], 2, 128), dtype=qkv.dtype, device=qkv.device)
        if qkv.shape[0] >= 16:
            _small_tiled_combined[(tr.cdiv(qkv.shape[0], 2), 13)](
                qkv,
                positions,
                freq,
                qk,
                attn_in,
                input_ids,
                ve_table,
                ve_gate,
                v,
                *qkv.stride(),
                positions.stride(0),
                freq.stride(0),
                *attn_in.stride(),
                input_ids.stride(0),
                *ve_table.stride(),
                *ve_gate.stride(),
                uses_rope,
                qkv.shape[0],
                num_warps=2,
                enable_fp_fusion=False,
            )
        else:
            _combined_prologue[(qkv.shape[0], 14)](
                qkv,
                positions,
                freq,
                qk,
                attn_in,
                input_ids,
                ve_table,
                ve_gate,
                v,
                *qkv.stride(),
                positions.stride(0),
                freq.stride(0),
                *attn_in.stride(),
                input_ids.stride(0),
                *ve_table.stride(),
                *ve_gate.stride(),
                uses_rope,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return qk, v
    qk = torch.empty((qkv.shape[0], 12, 128), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((qkv.shape[0], 2, 128), dtype=qkv.dtype, device=qkv.device)
    _large_combined_prologue[(tr.cdiv(qkv.shape[0], 4), 4)](
        qkv,
        positions,
        freq,
        qk,
        attn_in,
        input_ids,
        ve_table,
        ve_gate,
        v,
        *qkv.stride(),
        positions.stride(0),
        freq.stride(0),
        *attn_in.stride(),
        input_ids.stride(0),
        *ve_table.stride(),
        *ve_gate.stride(),
        uses_rope,
        qkv.shape[0],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return qk, v
