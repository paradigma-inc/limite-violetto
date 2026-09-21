"""Internal components for the single Limite vLLM execution graph.

The graph preserves four numerical invariants: fp32 logit softcapping, native
paged prefill and decode, an inclusive local-attention window, and ordered
left-to-right bfloat16 accumulation in the dense-residual mixer. Every model
weight is a registered parameter reachable through ``named_parameters()``.

Tensor and pipeline parallelism are rejected until their sharding rules are
implemented and verified.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.attention.backend import AttentionType


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """`limite_model.py:27-28`: no learnable gain, and no explicit epsilon.

    `eps=None` makes torch use `finfo(dtype).eps`, so the epsilon is
    dtype-dependent. Substituting a fixed `1e-6` changes the numbers, which is
    why `rms_norm_eps_mode` is a pinned config field rather than a default.
    """
    return F.rms_norm(x, (x.size(-1),))


def _param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    """An uninitialised registered parameter; no shape means a 0-dim scalar."""
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _ProjectionWeight(nn.Module):
    """Holds one `[out, in]` matrix at the checkpoint's `<name>.weight` path.

    A bare `nn.Linear` would work, but it also runs a Kaiming initialisation of
    every one of the 336 projection matrices at construction time and carries a
    `bias` slot this architecture does not have.
    """

    def __init__(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = _param(out_features, in_features, dtype=dtype)


class LimiteMuddMixer(nn.Module):
    """Dynamic dense residual mixing (`limite_model.py:381-395`).

    At the tapped layers the attention input is a learned, input-dependent
    weighted sum of a few earlier residual-stream states rather than the current
    one. `dense2` and `bias` are indexed by ABSOLUTE layer, so they are kept
    whole; the adapter's weight map preserves that (`spec.py:240-251`).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        num_layers = int(config.num_hidden_layers)
        taps = int(config.mudd_taps)
        inter = int(config.mudd_inter)
        hidden = int(config.hidden_size)
        # fp32 in the checkpoint, and the reference casts to the activation
        # dtype at the point of use rather than storing a narrowed copy.
        self.dense1 = _param(inter, hidden, dtype=torch.float32)
        self.dense2 = _param(num_layers, taps, inter, dtype=torch.float32)
        self.bias = _param(num_layers, taps, dtype=torch.float32)
        self.uses_r_way = bool(getattr(config, "mudd_mlp", False))
        if self.uses_r_way:
            # The R-way shares dense1 and reads the same history, but its
            # coefficients are a separately trained residual-base mixer.
            self.dense2_mlp = _param(num_layers, taps, inter, dtype=torch.float32)
            self.bias_mlp = _param(num_layers, taps, dtype=torch.float32)

    def combine(
        self, values: list[torch.Tensor], x_cur: torch.Tensor, layer_idx: int, *, r_way: bool = False
    ) -> torch.Tensor:
        """Ordered left-to-right accumulation, matching the trainer exactly.

        `limite_model.py:392-395` deliberately avoids materialising a stacked
        history: it seeds the accumulator with the first tap and adds the rest
        one at a time, in bfloat16. A single stacked contraction (an `einsum`
        over the tap axis) is mathematically identical and numerically is not,
        and `mudd_accumulation` is pinned in the config precisely so a
        reimplementation cannot quietly choose the other one.
        """
        count = len(values)
        inner = F.gelu(F.linear(rms_norm(x_cur), self.dense1.to(x_cur.dtype)))
        # einsum("btk,mk->btm", inner, dense2) is exactly this contraction.
        if r_way:
            if not self.uses_r_way:
                raise ValueError("MUDD R way requested by a config that carries no R-way tensors")
            dense2, bias = self.dense2_mlp, self.bias_mlp
        else:
            dense2, bias = self.dense2, self.bias
        weights = F.linear(inner, dense2[layer_idx, :count].to(inner.dtype))
        weights = weights + bias[layer_idx, :count].to(weights.dtype)
        out = weights[..., 0:1].type_as(values[0]) * values[0]
        for idx in range(1, count):
            out = out + weights[..., idx : idx + 1].type_as(values[idx]) * values[idx]
        return out


class LimiteRotary(nn.Module):
    """Partial, adjacent-pair-interleaved rotary with an odd-lane sign flip.

    `limite_model.py:199-216`. Only the first `2 * rope_n_pairs` dimensions of
    each head carry a non-zero frequency; the rest are padded with zeros and so
    pass through unrotated. `sin` is negated on the odd lanes and the rotation
    is applied against a pairwise-flipped copy of the input, which together give
    `(a, b) -> (a cos + b sin, b cos - a sin)`. The cos/sin tables are cast to
    bfloat16 before use, so their rounding is part of the trained function.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        head_dim = int(config.head_dim)
        n_pairs = int(config.rope_n_pairs)
        # rope_per_layer is false for this checkpoint, so local and global share
        # a base; the field is still read rather than assumed.
        if bool(config.rope_per_layer):
            raise NotImplementedError(
                "rope_per_layer=true would need a per-layer frequency table; this "
                "checkpoint declares false and the reference was verified for that."
            )
        base = float(config.rope_base_local)
        freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_pairs, dtype=torch.float32)
        freq = freq.repeat_interleave(2)
        freq = torch.cat([freq, freq.new_zeros(head_dim - 2 * n_pairs)])
        self.register_buffer("freq", freq, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.freq.numel()
        theta = positions.to(torch.float32).view(-1, 1) * self.freq.view(1, -1)
        cos = theta.cos().to(torch.bfloat16).view(-1, 1, head_dim)
        sin = theta.sin().to(torch.bfloat16).view(-1, 1, head_dim)
        sin[:, :, 1::2] *= -1
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
        return cos * x + sin * x_flip


def _per_layer_window(config: Any, layer_idx: int) -> int | None:
    """Return the runtime span in vLLM's convention.

    `limite_model.py:221-227` masks `k <= q` and `k >= q - window`, so a window
    of 1024 admits **1025** keys, the query token included. ``LimiteConfig``
    translates that serialized distance to a 1025-token runtime span. vLLM's backends
    subtract one before handing the number to FlashAttention
    (`vllm/v1/attention/backends/flash_attn.py:757-761`,
    `window_size = (sliding_window - 1, 0)`), so vLLM's number counts the query
    token, so the runtime value can be passed through unchanged.

    A negative window means unlimited (`limite_model.py:225`), which vLLM spells
    as `None`.
    """
    window = (
        int(config.global_window)
        if layer_idx in {int(x) for x in config.global_layers}
        else int(config.sliding_window)
    )
    return None if window < 0 else window


class _AttentionBase(nn.Module):
    """One attention branch: projections, value embeddings, rotary, paged
    attention, the cross-head correction, and the output projection.

    Covers `limite_model.py:307-338` (`_ve`, `_project_qkv`) and `:344-354` (the
    attention half of `_finish_layer`).
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        rotary: LimiteRotary,
        cache_config: Any,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.gqa_groups = self.num_heads // self.num_kv_heads
        self.ve_dim = int(config.ve_dim)
        self.ve_stored_heads = int(getattr(config, "ve_stored_heads", self.num_heads))
        self.ve_gate_scale = float(config.ve_gate_scale)
        self.attn_gate_channels = int(config.attn_gate_channels)
        self.attn_gate_scale = float(getattr(config, "attn_gate_scale", 2.0))
        self.xsa_eps = float(config.xsa_normalize_eps)
        self.uses_ve = layer_idx in {int(x) for x in config.ve_layers}
        self.uses_xsa = bool(config.xsa) and layer_idx in {
            int(x) for x in config.xsa_layers
        }
        self.is_global = layer_idx in {int(x) for x in config.global_layers}
        # `global_nope: true` -- global layers skip rotary entirely
        # (limite_model.py:333-335).
        self.uses_rope = not (bool(config.global_nope) and self.is_global)
        self.rotary = rotary

        weight_dtype = torch.get_default_dtype()
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = _ProjectionWeight(self.hidden_size, self.hidden_size, weight_dtype)
        self.k_proj = _ProjectionWeight(kv_size, self.hidden_size, weight_dtype)
        self.v_proj = _ProjectionWeight(kv_size, self.hidden_size, weight_dtype)
        self.o_proj = _ProjectionWeight(self.hidden_size, self.hidden_size, weight_dtype)

        # Per-layer learned scalars, folded into the projections at runtime
        # (limite_model.py:323-326 and :354). Stored fp32, as exported, but see
        # `_fold` -- the fold itself runs in bf16. Folding once at load time
        # would be bit-identical but would leave the folded value in the
        # parameter, so any weight-synchronisation path that writes `param.data`
        # without going through `load_weights` would silently double-scale.
        # Runtime folding has no such failure mode.
        self.qkv_scale = _param(dtype=torch.float32)
        self.o_scale = _param(dtype=torch.float32)
        self.xsa_alpha = _param(self.num_heads, dtype=torch.float32)
        if self.uses_ve:
            self.ve_gate = _param(
                self.ve_stored_heads, int(config.ve_gate_channels), dtype=torch.float32
            )
        if self.attn_gate_channels:
            self.attn_gate = _param(self.num_heads, self.attn_gate_channels, dtype=torch.float32)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            float(config.attention_softmax_scale),
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=_per_layer_window(config, layer_idx),
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def _apply_value_embeddings(
        self,
        input_ids: torch.Tensor,
        attn_in: torch.Tensor,
        v: torch.Tensor,
        value_embeds: VocabParallelEmbedding,
    ) -> torch.Tensor:
        """`limite_model.py:307-318`.

        Tables may be query-width (v4) or key-value-width (v5). The config
        records which tensor layout was exported, so the view is never guessed.
        """
        ve = value_embeds(input_ids).to(v.dtype).view(-1, self.ve_stored_heads, self.ve_dim)
        ve = F.pad(ve, (0, self.head_dim - self.ve_dim))
        gate_w = self.ve_gate.to(attn_in.dtype)
        if self.ve_stored_heads > self.num_kv_heads:
            ve = ve[:, : self.num_kv_heads]
            gate_w = gate_w[: self.num_kv_heads]
        gate = self.ve_gate_scale * torch.sigmoid(
            F.linear(attn_in[..., : gate_w.size(-1)], gate_w)
        )
        return v + gate.unsqueeze(-1) * ve

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_in: torch.Tensor,
        value_embeds: VocabParallelEmbedding,
        attn_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = F.linear(attn_in, self._fold(self.qkv_scale, self.q_proj.weight))
        k = F.linear(attn_in, self._fold(self.qkv_scale, self.k_proj.weight))
        v = F.linear(attn_in, self._fold(self.qkv_scale, self.v_proj.weight))
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.uses_ve:
            # The VE-adjusted value tensor is what attention consumes AND what
            # is written to the paged KV cache: `_layer_prefill` caches the
            # post-VE `v` (limite_model.py:330-331, :424-426). Caching the raw
            # projection instead is a documented upstream failure.
            v = self._apply_value_embeddings(input_ids, attn_in, v, value_embeds)

        # QK norm happens after the value-embedding step and before rotary
        # (`qk_norm: rms_pre_rope`, limite_model.py:332).
        q, k = rms_norm(q), rms_norm(k)
        if self.uses_rope:
            q, k = self.rotary(q, k, positions)

        y = self.attn(q, k, v).view(-1, self.num_heads, self.head_dim)

        if self.uses_xsa:
            # limite_model.py:344-348, fp32 throughout.
            vn = F.normalize(self._expand_kv(v).float(), dim=-1, eps=self.xsa_eps)
            proj = (y.float() * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(self.xsa_alpha.float()).view(1, self.num_heads, 1)
            y = y - (alpha * proj * vn).to(y.dtype)

        if self.attn_gate_channels:
            gate = self.attn_gate_scale * torch.sigmoid(
                F.linear(attn_in[..., : self.attn_gate_channels], self.attn_gate.to(attn_in.dtype))
            )
            y = y * gate.to(y.dtype).unsqueeze(-1)

        y = y.contiguous().view(-1, self.hidden_size)
        return F.linear(y, self._fold(self.o_scale, self.o_proj.weight))

    @staticmethod
    def _fold(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Fold a learned scalar into a projection, in the weight's dtype.

        The cast is written out rather than left to type promotion, and that is
        not cosmetic. The reference writes `sa[0] * weight` where `sa[0]` is a
        **0-dim** fp32 tensor and `weight` is bf16 (`limite_model.py:324-326`,
        `:354`). PyTorch treats a 0-dim tensor as a wrapped scalar for promotion,
        so the result is **bf16, not fp32**: the scalar is rounded down to bf16
        before it ever multiplies anything, and the fp32 precision the checkpoint
        stores for these six scalars is never used by anyone.

        Verified on this checkpoint: rounding `scalars` to bf16 inside the oracle
        shifts its log-probabilities by exactly 0.0 (against a live-knob control
        where a 1% scalar change shifts them by 5.8e-02), and the folded bf16
        products are identical across all 1,638,400 elements of a q_proj.

        The hazard this makes explicit: the same expression with a **1-element**
        fp32 tensor promotes to fp32 and gives different numbers (measured: 4 of
        32 elements differ on a toy case). FSDP2 rejects 0-dim parameters, so the
        trainer is forced to hold these as 1-element vectors and restores 0-dim
        with `.view(())` to stay on this path. Anything here that turned
        `qkv_scale` into shape `(1,)` would silently switch the fold to fp32 and
        diverge from both the oracle and the trainer, with no error and no
        shape mismatch. `tests/test_limite_vllm_model.py` pins it.
        """
        return (scale.to(weight.dtype) * weight)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.num_heads:
            return x.repeat_interleave(self.gqa_groups, dim=1)
        return x


class _MLPBase(nn.Module):
    """The declared relu2 or SwiGLU feed-forward family.

    `down_proj.weight` is the adapter's transposition of a matrix the reference
    consumes as a bare `@` contraction while its sibling goes through
    `F.linear`; after transposition both are HF-oriented `[out, in]`
    (`spec.py:131-141`).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        inter = int(config.intermediate_size)
        dtype = torch.get_default_dtype()
        self.up_proj = _ProjectionWeight(inter, hidden, dtype)
        self.down_proj = _ProjectionWeight(hidden, inter, dtype)
        self.mlp_type = str(config.mlp_type)
        if self.mlp_type == "swiglu":
            self.gate_proj = _ProjectionWeight(inter, hidden, dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        up = F.linear(h, self.up_proj.weight)
        if self.mlp_type == "relu2":
            activated = F.relu(up).square()
        elif self.mlp_type == "swiglu":
            activated = F.silu(F.linear(h, self.gate_proj.weight)) * up
        else:  # Configuration validation rejects this before construction.
            raise AssertionError(f"unsupported mlp_type {self.mlp_type!r}")
        return F.linear(activated, self.down_proj.weight)


class _DecoderLayerBase(nn.Module):
    """One block. There is no plain additive residual anywhere: both joins are
    `resid_lambda * stream + post_lambda * branch` with learned coefficients
    (`limite_model.py:355-357` and `:366-368`).
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        rotary: LimiteRotary,
        cache_config: Any,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = _AttentionBase(
            config,
            layer_idx,
            rotary,
            cache_config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        self.mlp = _MLPBase(config)
        self.resid_lambda_attn = _param(dtype=torch.float32)
        self.post_lambda_attn = _param(dtype=torch.float32)
        self.resid_lambda_mlp = _param(dtype=torch.float32)
        self.post_lambda_mlp = _param(dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        x: torch.Tensor,
        attn_in: torch.Tensor,
        value_embeds: VocabParallelEmbedding,
        attn_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(input_ids, positions, attn_in, value_embeds)
        # `.to(x.dtype)` first: the reference rounds the fp32 coefficient down to
        # the activation dtype before multiplying (limite_model.py:355-356).
        residual_base = x if attn_residual is None else attn_residual
        x = (
            self.resid_lambda_attn.to(x.dtype) * residual_base
            + self.post_lambda_attn.to(x.dtype) * attn_out
        )
        mlp_out = self.mlp(rms_norm(x))
        return (
            self.resid_lambda_mlp.to(x.dtype) * x
            + self.post_lambda_mlp.to(x.dtype) * mlp_out
        )
