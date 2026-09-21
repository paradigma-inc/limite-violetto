"""The single supported Limite execution graph for vLLM.

QKV and SwiGLU projections are fused, and projection scalars are folded once
after every exhaustive weight load.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.attention.backend import AttentionType

from limite_vllm.validation import validate as validate_config
from limite_vllm._components import (
    _AttentionBase,
    _DecoderLayerBase,
    _MLPBase,
    LimiteMuddMixer,
    LimiteRotary,
    _ProjectionWeight,
    _per_layer_window,
    _param,
    rms_norm,
)


def _stacked_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor, row_start: int
) -> None:
    """Load one logical projection into a row slice of a fused parameter."""
    destination = param.narrow(0, row_start, loaded_weight.shape[0])
    default_weight_loader(destination, loaded_weight)


class LimiteAttention(_AttentionBase):
    """Attention with one fused QKV GEMM and pre-folded projection scales."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        rotary: LimiteRotary,
        cache_config: Any,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.gqa_groups = self.num_heads // self.num_kv_heads
        self.ve_dim = int(config.ve_dim)
        self.ve_stored_heads = int(
            getattr(config, "ve_stored_heads", self.num_heads)
        )
        self.ve_gate_scale = float(config.ve_gate_scale)
        self.attn_gate_channels = int(config.attn_gate_channels)
        self.attn_gate_scale = float(getattr(config, "attn_gate_scale", 2.0))
        self.xsa_eps = float(config.xsa_normalize_eps)
        self.uses_ve = layer_idx in {int(x) for x in config.ve_layers}
        self.uses_xsa = bool(config.xsa) and layer_idx in {
            int(x) for x in config.xsa_layers
        }
        self.is_global = layer_idx in {int(x) for x in config.global_layers}
        self.uses_rope = not (bool(config.global_nope) and self.is_global)
        self.rotary = rotary

        self.q_size = self.hidden_size
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = _ProjectionWeight(
            self.q_size + 2 * self.kv_size,
            self.hidden_size,
            torch.get_default_dtype(),
        )
        self.qkv_proj.weight.weight_loader = _stacked_weight_loader
        self.o_proj = _ProjectionWeight(
            self.hidden_size, self.hidden_size, torch.get_default_dtype()
        )
        self.qkv_scale = _param(dtype=torch.float32)
        self.o_scale = _param(dtype=torch.float32)
        self.xsa_alpha = _param(self.num_heads, dtype=torch.float32)
        if self.uses_ve:
            self.ve_gate = _param(
                self.ve_stored_heads,
                int(config.ve_gate_channels),
                dtype=torch.float32,
            )
        if self.attn_gate_channels:
            self.attn_gate = _param(
                self.num_heads,
                self.attn_gate_channels,
                dtype=torch.float32,
            )
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_in: torch.Tensor,
        value_embeds: VocabParallelEmbedding,
        attn_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = F.linear(attn_in, self.qkv_proj.weight)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.uses_ve:
            v = self._apply_value_embeddings(input_ids, attn_in, v, value_embeds)

        q, k = rms_norm(q), rms_norm(k)
        if self.uses_rope:
            q, k = self.rotary(q, k, positions)

        y = self.attn(q, k, v).view(-1, self.num_heads, self.head_dim)
        if self.uses_xsa:
            vn = F.normalize(self._expand_kv(v).float(), dim=-1, eps=self.xsa_eps)
            proj = (y.float() * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(self.xsa_alpha.float()).view(1, self.num_heads, 1)
            y = y - (alpha * proj * vn).to(y.dtype)
        if self.attn_gate_channels:
            gate = self.attn_gate_scale * torch.sigmoid(
                F.linear(
                    attn_in[..., : self.attn_gate_channels],
                    self.attn_gate.to(attn_in.dtype),
                )
            )
            y = y * gate.to(y.dtype).unsqueeze(-1)

        y = y.contiguous().view(-1, self.hidden_size)
        return F.linear(y, self.o_proj.weight)

    @torch.no_grad()
    def fold_projection_scales_(self) -> None:
        """Materialize the exact weight-dtype folds after raw weights load."""
        self.qkv_proj.weight.mul_(self.qkv_scale.to(self.qkv_proj.weight.dtype))
        self.o_proj.weight.mul_(self.o_scale.to(self.o_proj.weight.dtype))


class LimiteMLP(_MLPBase):
    """SwiGLU with a single gate/up GEMM; ReLU2 remains unchanged."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if self.mlp_type == "swiglu":
            del self.up_proj
            del self.gate_proj
            self.intermediate_size = int(config.intermediate_size)
            self.gate_up_proj = _ProjectionWeight(
                2 * self.intermediate_size,
                int(config.hidden_size),
                torch.get_default_dtype(),
            )
            self.gate_up_proj.weight.weight_loader = _stacked_weight_loader

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "relu2":
            activated = F.relu(F.linear(h, self.up_proj.weight)).square()
        elif self.mlp_type == "swiglu":
            gate, up = F.linear(h, self.gate_up_proj.weight).split(
                self.intermediate_size, dim=-1
            )
            activated = F.silu(gate) * up
        else:
            raise AssertionError(f"unsupported mlp_type {self.mlp_type!r}")
        return F.linear(activated, self.down_proj.weight)


class LimiteDecoderLayer(_DecoderLayerBase):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        rotary: LimiteRotary,
        cache_config: Any,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.self_attn = LimiteAttention(
            config,
            layer_idx,
            rotary,
            cache_config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        self.mlp = LimiteMLP(config)
        self.resid_lambda_attn = _param(dtype=torch.float32)
        self.post_lambda_attn = _param(dtype=torch.float32)
        self.resid_lambda_mlp = _param(dtype=torch.float32)
        self.post_lambda_mlp = _param(dtype=torch.float32)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "tokens"},
        "positions": {0: "tokens"},
    }
)
class LimiteModel(nn.Module):
    """Compiled Limite model body."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        validate_config(config)
        self.config = config
        if getattr(cache_config, "sliding_window", None) is not None:
            raise RuntimeError(
                f"CacheConfig.sliding_window is {cache_config.sliding_window}, but "
                "Limite interleaves local and unwindowed global layers."
            )

        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            int(config.hidden_size),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.value_embeds = VocabParallelEmbedding(
            int(config.vocab_size),
            int(getattr(config, "ve_stored_heads", config.num_attention_heads))
            * int(config.ve_dim),
            prefix=maybe_prefix(prefix, "value_embeds"),
        )
        self.rotary = LimiteRotary(config)
        self.mudd = LimiteMuddMixer(config)
        self.layers = nn.ModuleList(
            [
                LimiteDecoderLayer(
                    config,
                    index,
                    self.rotary,
                    cache_config,
                    prefix=maybe_prefix(prefix, f"layers.{index}"),
                )
                for index in range(int(config.num_hidden_layers))
            ]
        )
        self.mudd_tap_idx: dict[int, list[int]] = {
            int(layer): [int(index) for index in taps]
            for layer, taps in dict(config.mudd_tap_idx).items()
        }
        if set(self.mudd_tap_idx) != {int(x) for x in config.mudd_layers}:
            raise ValueError(
                f"mudd_tap_idx covers {sorted(self.mudd_tap_idx)} but mudd_layers "
                f"is {sorted(int(x) for x in config.mudd_layers)}."
            )
        self.retained_history = {
            index for taps in self.mudd_tap_idx.values() for index in taps
        }
        self.final_softcap = float(config.final_softcap)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = rms_norm(self.embed_tokens(input_ids))
        history: dict[int, torch.Tensor] = {0: x} if self.mudd_tap_idx else {}
        for index, layer in enumerate(self.layers):
            if index in self.mudd_tap_idx:
                values = [history[i] for i in self.mudd_tap_idx[index]]
                attn_in = rms_norm(self.mudd.combine(values, x, index))
                residual_base = (
                    self.mudd.combine(values, x, index, r_way=True)
                    if self.mudd.uses_r_way
                    else x
                )
            else:
                attn_in = rms_norm(x)
                residual_base = x
            x = layer(
                input_ids,
                positions,
                x,
                attn_in,
                self.value_embeds,
                residual_base,
            )
            if index + 1 in self.retained_history:
                history[index + 1] = x
        if self.final_softcap > 0:
            cap = self.final_softcap
            x = cap * torch.tanh(x / cap)
        return rms_norm(x)


class LimiteForCausalLM(nn.Module):
    """vLLM entry point for ``architectures: ["LimiteForCausalLM"]``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
            raise NotImplementedError("Limite modeling supports data parallelism only.")
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = LimiteModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = self.model.embed_tokens
        softcap = dict(config.softcap_logits)
        if softcap.get("kind") != "sigmoid":
            raise NotImplementedError(
                f"softcap_logits.kind={softcap.get('kind')!r}; only sigmoid is supported."
            )
        self.softcap_a = float(softcap["a"])
        self.softcap_b = float(softcap["b"])
        self.softcap_c = float(softcap["c"])
        self.head_precision_mode = str(config.lm_head_precision_mode)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            raise NotImplementedError(
                "Limite reads token ids directly on every value-embedding layer, "
                "so an embeddings-only input cannot reconstruct the value stream."
            )
        if intermediate_tensors is not None:
            raise NotImplementedError("Limite does not support pipeline parallelism.")
        if input_ids is None:
            raise ValueError("Limite requires input_ids.")
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the softcapped head at the configured precision."""
        weight = self.lm_head.weight
        if self.head_precision_mode == "oracle_exact":
            raw = F.linear(hidden_states, weight.to(hidden_states.dtype))
        elif hidden_states.is_cuda and hidden_states.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            raw = torch.mm(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                weight.t(),
                out_dtype=torch.float32,
            )
        else:
            raw = F.linear(hidden_states.float(), weight.float())
        return self.softcap_a * torch.sigmoid(
            (raw.float() + self.softcap_b) / self.softcap_c
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        source_map: dict[str, tuple[str, torch.Tensor, int | None]] = {}
        for param_name, param in params.items():
            if param_name.endswith(".self_attn.qkv_proj.weight"):
                stem = param_name.removesuffix("qkv_proj.weight")
                kv_size = int(self.config.num_key_value_heads) * int(
                    self.config.head_dim
                )
                source_map[stem + "q_proj.weight"] = (param_name, param, 0)
                source_map[stem + "k_proj.weight"] = (
                    param_name,
                    param,
                    int(self.config.hidden_size),
                )
                source_map[stem + "v_proj.weight"] = (
                    param_name,
                    param,
                    int(self.config.hidden_size) + kv_size,
                )
            elif param_name.endswith(".mlp.gate_up_proj.weight"):
                stem = param_name.removesuffix("gate_up_proj.weight")
                source_map[stem + "gate_proj.weight"] = (param_name, param, 0)
                source_map[stem + "up_proj.weight"] = (
                    param_name,
                    param,
                    int(self.config.intermediate_size),
                )
            else:
                source_map[param_name] = (param_name, param, None)

        loaded_sources: set[str] = set()
        loaded_params: set[str] = set()
        unexpected: list[str] = []
        for name, weight in weights:
            if name == "lm_head.weight":
                name = "model.embed_tokens.weight"
            target = source_map.get(name)
            if target is None:
                unexpected.append(name)
                continue
            target_name, param, row_start = target
            loader = getattr(param, "weight_loader", default_weight_loader)
            if row_start is None:
                loader(param, weight)
            else:
                loader(param, weight, row_start)
            loaded_sources.add(name)
            loaded_params.add(target_name)

        missing = sorted(set(source_map) - loaded_sources)
        if unexpected or missing:
            raise ValueError(
                "Limite weight loading is not a bijection against the "
                f"checkpoint. Unexpected ({len(unexpected)}): "
                f"{sorted(unexpected)[:10]}. Never loaded ({len(missing)}): "
                f"{missing[:10]}."
            )
        for layer in self.model.layers:
            layer.self_attn.fold_projection_scales_()
        return loaded_params
