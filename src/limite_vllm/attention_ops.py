"""Limite's attention prologue and epilogue as ``torch.ops.limite`` custom ops.

Everything between the fused QKV GEMM and paged attention (value embeddings,
gain-free QK RMS norm, partial interleaved rotary) is one Triton launch, and so
is everything between attention and ``o_proj`` (XSA correction and the
``2 * sigmoid`` output gate). The kernels in :mod:`limite_vllm.triton_prologue`
and :mod:`limite_vllm.triton_epilogue` reproduce every bf16 rounding of the
torch graph in :mod:`limite_vllm._components` and hard-code the attention shape
that :func:`limite_vllm.validation.validate` admits.

Each launch is a functional custom op: vLLM's ``torch.compile`` keeps it opaque,
its shape dispatch runs on concrete sizes, and its CUDA graphs capture the
launch. The ops return fresh tensors only; Q and K are views of one
``[T, 12, 128]`` buffer, and a layer without value embeddings passes V through
as a view of the QKV GEMM output, bit for bit.
"""

import torch
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op

from limite_vllm import triton_epilogue, triton_prologue

# The ops live as long as this library object, i.e. for the process.
_LIBRARY = Library("limite", "FRAGMENT")


def _attention_prologue(
    qkv: torch.Tensor, positions: torch.Tensor, freq: torch.Tensor, uses_rope: bool
) -> torch.Tensor:
    return triton_prologue.prologue(qkv, positions, freq, uses_rope=uses_rope)


def _attention_prologue_fake(
    qkv: torch.Tensor, positions: torch.Tensor, freq: torch.Tensor, uses_rope: bool
) -> torch.Tensor:
    return qkv.new_empty((qkv.shape[0], 12, 128))


def _attention_prologue_ve(
    qkv: torch.Tensor,
    attn_in: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    ve_table: torch.Tensor,
    ve_gate: torch.Tensor,
    freq: torch.Tensor,
    uses_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return triton_prologue.prologue_ve(
        qkv, attn_in, positions, input_ids, ve_table, ve_gate, freq, uses_rope=uses_rope
    )


def _attention_prologue_ve_fake(
    qkv: torch.Tensor,
    attn_in: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    ve_table: torch.Tensor,
    ve_gate: torch.Tensor,
    freq: torch.Tensor,
    uses_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    return qkv.new_empty((tokens, 12, 128)), qkv.new_empty((tokens, 2, 128))


def _attention_epilogue(
    y: torch.Tensor,
    v: torch.Tensor,
    attn_in: torch.Tensor,
    xsa_alpha: torch.Tensor,
    attn_gate: torch.Tensor,
) -> torch.Tensor:
    return triton_epilogue.epilogue(y, v, attn_in, xsa_alpha, attn_gate)


def _attention_epilogue_fake(
    y: torch.Tensor,
    v: torch.Tensor,
    attn_in: torch.Tensor,
    xsa_alpha: torch.Tensor,
    attn_gate: torch.Tensor,
) -> torch.Tensor:
    return y.new_empty((y.shape[0], 1280))


for _name, _impl, _fake in (
    ("attention_prologue", _attention_prologue, _attention_prologue_fake),
    ("attention_prologue_ve", _attention_prologue_ve, _attention_prologue_ve_fake),
    ("attention_epilogue", _attention_epilogue, _attention_epilogue_fake),
):
    direct_register_custom_op(
        _name, _impl, mutates_args=[], fake_impl=_fake, target_lib=_LIBRARY
    )
