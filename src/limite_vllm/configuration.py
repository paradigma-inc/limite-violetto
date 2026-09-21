"""Configuration contract and validation for Limite checkpoints in vLLM.

Every numerical choice is explicit in ``config.json``. Unsupported values fail
at load time instead of silently selecting different model behavior.

Field semantics are documented in the artifact's own `config_field_notes.json`.
"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from limite_vllm.contract import (
    ARCHITECTURE,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MODEL_TYPE,
    PAD_TOKEN_ID,
)

REQUIRED_CONFIG_FIELDS: tuple[str, ...] = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "padded_vocab_size",
    "tokenizer_vocab_size",
    "max_position_embeddings",
    "tie_word_embeddings",
    "torch_dtype",
    "rms_norm_has_weight",
    "rms_norm_eps_mode",
    "qk_norm",
    "attention_softmax_scale",
    "sliding_window",
    "sliding_window_convention",
    "global_window",
    "global_layers",
    "global_every",
    "global_nope",
    "attn_gate_channels",
    "attn_gate_scale",
    "attn_gate_applied",
    "pos_mode",
    "rope_frac",
    "rope_base_local",
    "rope_base_global",
    "rope_per_layer",
    "rope_n_pairs",
    "rope_style",
    "rope_cos_sin_dtype",
    "ve_dim",
    "ve_layers",
    "ve_gate_channels",
    "ve_gate_scale",
    "ve_head_slice",
    "ve_stored_heads",
    "ve_applied_before_qk_norm",
    "xsa",
    "xsa_layers",
    "xsa_normalize_eps",
    "mudd",
    "mudd_at",
    "mudd_layers",
    "mudd_taps",
    "mudd_inter",
    "mudd_tap_idx",
    "mudd_mlp",
    "mudd_hist_convention",
    "mudd_accumulation",
    "mlp_type",
    "mlp_formula",
    "mlp_ratio",
    "softcap_logits",
    "final_softcap",
    "lm_head_precision_mode",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "source_format",
    "checkpoint_step",
    "checkpoint_world_size",
)

HEAD_PRECISION_MODES: tuple[str, ...] = ("oracle_exact", "fp32_accumulate")

#: Values the modeling code implements. Anything else must fail loudly: every
#: entry here is a fork in the numerics, and a silent fallback would produce a
#: plausible-looking model with wrong numbers.
SUPPORTED = {
    "mlp_type": {"relu2", "swiglu"},
    "pos_mode": {"rope"},
    "qk_norm": {"rms_pre_rope"},
    "rms_norm_eps_mode": {"torch_finfo_default"},
    "rope_style": {"interleaved_pairs_odd_lane_sign_flip"},
    "rope_cos_sin_dtype": {"bfloat16"},
    "mudd_accumulation": {"ordered_left_to_right"},
    "ve_head_slice": {"first_num_key_value_heads"},
    "lm_head_precision_mode": set(HEAD_PRECISION_MODES),
    "softcap_kind": {"sigmoid"},
}

MLP_FORMULAS = {
    "relu2": "relu(fc(h)) ** 2 @ down_proj",
    "swiglu": "(silu(gate_proj(h)) * up_proj(h)) @ down_proj, no clamp on either factor",
}

#: The reference applies `k >= q - sliding_window`, i.e. the window is inclusive
#: of the query token, so a local layer sees `sliding_window + 1` keys. Matching
#: the raw number against an exclusive-convention kernel silently drops the
#: oldest key on every local layer.
SLIDING_WINDOW_CONVENTION = "k >= q - sliding_window, inclusive of the query token (span = sliding_window + 1 keys)"


def _derive_layer_types(num_hidden_layers: int, global_layers: list[int]) -> list[str]:
    global_layer_set = set(global_layers)
    return [
        "full_attention" if layer_idx in global_layer_set else "sliding_attention"
        for layer_idx in range(num_hidden_layers)
    ]


class LimiteConfig(PretrainedConfig):
    """Hugging Face configuration object used while vLLM loads Limite."""

    model_type = MODEL_TYPE
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "LimiteConfig":
        missing = [
            field
            for field in REQUIRED_CONFIG_FIELDS
            if (
                config_dict.get("torch_dtype", config_dict.get("dtype"))
                if field == "torch_dtype"
                else config_dict.get(field)
            )
            is None
        ]
        if missing:
            raise ValueError(
                f"Limite config is missing required fields {missing}. Re-export the checkpoint "
                "instead of guessing numerical choices."
            )
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        # -- shape ------------------------------------------------------------
        vocab_size: int = 151680,
        padded_vocab_size: int | None = None,
        tokenizer_vocab_size: int | None = None,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 10,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        mlp_ratio: int = 4,
        mlp_type: str = "relu2",
        mlp_formula: str | None = None,
        max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,
        # -- normalisation ----------------------------------------------------
        rms_norm_has_weight: bool = False,
        rms_norm_eps_mode: str = "torch_finfo_default",
        qk_norm: str = "rms_pre_rope",
        # -- attention --------------------------------------------------------
        attention_softmax_scale: float = 0.1,
        sliding_window: int = 1024,
        sliding_window_convention: str = SLIDING_WINDOW_CONVENTION,
        global_window: int = -1,
        global_layers: list[int] | None = None,
        global_every: int = 4,
        global_nope: bool = True,
        attn_gate_channels: int = 0,
        attn_gate_scale: float = 2.0,
        attn_gate_applied: str = "per_head_before_o_proj",
        attention_dropout: float = 0.0,
        # -- rotary -----------------------------------------------------------
        pos_mode: str = "rope",
        rope_frac: float = 0.5,
        rope_base_local: float = 1024.0,
        rope_base_global: float = 1024.0,
        rope_per_layer: bool = False,
        rope_n_pairs: int | None = None,
        rope_style: str = "interleaved_pairs_odd_lane_sign_flip",
        rope_cos_sin_dtype: str = "bfloat16",
        # -- value embeddings -------------------------------------------------
        ve_dim: int = 128,
        ve_layers: list[int] | None = None,
        ve_gate_channels: int = 12,
        ve_gate_scale: float = 2.0,
        ve_head_slice: str = "first_num_key_value_heads",
        ve_stored_heads: int | None = None,
        ve_applied_before_qk_norm: bool = True,
        # -- cross-head attention correction ----------------------------------
        xsa: bool = True,
        xsa_layers: list[int] | None = None,
        xsa_normalize_eps: float = 1e-4,
        # -- dynamic dense residual mixing ------------------------------------
        mudd: bool = True,
        mudd_at: list[int] | None = None,
        mudd_layers: list[int] | None = None,
        mudd_taps: int = 3,
        mudd_inter: int = 32,
        mudd_tap_idx: dict[str, list[int]] | None = None,
        mudd_hist_convention: str = "hist[0] = rms_norm(embedding); hist[j] = output of layer j-1",
        mudd_accumulation: str = "ordered_left_to_right",
        mudd_mlp: bool = False,
        mudd_r_site: str | None = None,
        # -- output head ------------------------------------------------------
        softcap_logits: dict[str, Any] | None = None,
        final_softcap: float = 0.0,
        lm_head_precision_mode: str = "oracle_exact",
        lm_head_compute_dtype: str | None = None,  # legacy; refused below, never reinterpreted
        # -- tokens -----------------------------------------------------------
        pad_token_id: int | None = PAD_TOKEN_ID,
        bos_token_id: int | None = BOS_TOKEN_ID,
        eos_token_id: int | list[int] | None = EOS_TOKEN_ID,
        # -- provenance (carried through, never read by the model) ------------
        source_format: str | None = None,
        source_metadata_assumptions: list[str] | None = None,
        checkpoint_step: int | None = None,
        checkpoint_world_size: int | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("architectures", [ARCHITECTURE])
        # Keep the checkpoint contract explicit even though vLLM supplies the
        # optimized execution graph.
        kwargs.setdefault("attn_implementation", "eager")
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size if padded_vocab_size is not None else vocab_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self.mlp_type = mlp_type
        self.mlp_formula = mlp_formula if mlp_formula is not None else MLP_FORMULAS.get(mlp_type)
        self.max_position_embeddings = max_position_embeddings

        self.rms_norm_has_weight = rms_norm_has_weight
        self.rms_norm_eps_mode = rms_norm_eps_mode
        self.qk_norm = qk_norm

        self.attention_softmax_scale = attention_softmax_scale
        # Checkpoints store the maximum *distance* to a visible key. vLLM's
        # attention backends consume the total visible span, including the
        # current token. Preserve the serialized value for round trips while
        # exposing the backend span at runtime.
        self._serialized_sliding_window = int(sliding_window)
        self.sliding_window = self._serialized_sliding_window + 1
        self.sliding_window_convention = sliding_window_convention
        self.global_window = global_window
        self.global_every = global_every
        self.global_layers = sorted(int(x) for x in (global_layers or []))
        self.global_nope = global_nope
        self.attn_gate_channels = attn_gate_channels
        self.attn_gate_scale = attn_gate_scale
        self.attn_gate_applied = attn_gate_applied
        self.attention_dropout = attention_dropout

        self.pos_mode = pos_mode
        self.rope_frac = rope_frac
        self.rope_base_local = rope_base_local
        self.rope_base_global = rope_base_global
        self.rope_per_layer = rope_per_layer
        self.rope_n_pairs = rope_n_pairs if rope_n_pairs is not None else max(1, int(head_dim * rope_frac) // 2)
        self.rope_style = rope_style
        self.rope_cos_sin_dtype = rope_cos_sin_dtype

        self.ve_dim = ve_dim
        self.ve_layers = sorted(int(x) for x in (ve_layers or []))
        self.ve_gate_channels = ve_gate_channels
        self.ve_gate_scale = ve_gate_scale
        self.ve_head_slice = ve_head_slice
        self.ve_stored_heads = ve_stored_heads if ve_stored_heads is not None else num_attention_heads
        self.ve_applied_before_qk_norm = ve_applied_before_qk_norm

        self.xsa = xsa
        self.xsa_layers = sorted(int(x) for x in (xsa_layers or []))
        self.xsa_normalize_eps = xsa_normalize_eps

        self.mudd = mudd
        self.mudd_at = sorted(int(x) for x in (mudd_at or []))
        self.mudd_layers = sorted(int(x) for x in (mudd_layers if mudd_layers is not None else self.mudd_at))
        self.mudd_taps = mudd_taps
        self.mudd_inter = mudd_inter
        self.mudd_tap_idx = {str(k): [int(i) for i in v] for k, v in (mudd_tap_idx or {}).items()}
        self.mudd_hist_convention = mudd_hist_convention
        self.mudd_accumulation = mudd_accumulation
        self.mudd_mlp = mudd_mlp
        self.mudd_r_site = (mudd_r_site or "resid") if mudd_mlp else None

        self.softcap_logits = dict(softcap_logits or {"kind": "sigmoid", "a": 23.0, "b": 5.0, "c": 7.5})
        self.final_softcap = final_softcap
        # The legacy field named a dtype, which was ambiguous: the oracle has three
        # roundable stages and "float32" did not say which, so two independent
        # implementations read it two different ways. A config carrying it is
        # refused rather than mapped, so a stale checkpoint cannot quietly land on
        # a default that happens to agree with the engine. Bare construction is
        # still allowed -- transformers requires config classes to build with no
        # arguments -- and defaults to the only mode either side implements.
        # The vLLM graph consumes this field directly.
        if lm_head_compute_dtype is not None:
            raise ValueError(
                f"Limite config carries the superseded lm_head_compute_dtype={lm_head_compute_dtype!r}. "
                "That field named only one of the head's three roundable stages and is not "
                "reinterpreted; re-export the checkpoint."
            )
        self.lm_head_precision_mode = lm_head_precision_mode

        self.source_format = source_format
        self.source_metadata_assumptions = source_metadata_assumptions
        self.checkpoint_step = checkpoint_step
        self.checkpoint_world_size = checkpoint_world_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if getattr(self, "layer_types", None) is None:
            self.layer_types = _derive_layer_types(self.num_hidden_layers, self.global_layers)

        self.validate_architecture()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint convention, not the native cache span."""
        output = super().to_dict()
        output["sliding_window"] = self._serialized_sliding_window
        output.pop("_serialized_sliding_window", None)
        return output

    # -- derived ---------------------------------------------------------------

    @property
    def ve_layer_to_ordinal(self) -> dict[int, int]:
        """`ve_gate` is exported indexed by a compacted ve-layer ordinal; the
        adapter already unpacked it onto absolute layer indices, so the model
        only needs membership. Kept for the conversion chain's benefit."""
        return {layer: ordinal for ordinal, layer in enumerate(self.ve_layers)}

    def tap_indices(self, layer_idx: int) -> list[int] | None:
        return self.mudd_tap_idx.get(str(layer_idx))

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx in set(self.global_layers)

    # -- validation ------------------------------------------------------------

    def validate_architecture(self) -> None:
        for field, allowed in (
            ("mlp_type", SUPPORTED["mlp_type"]),
            ("pos_mode", SUPPORTED["pos_mode"]),
            ("qk_norm", SUPPORTED["qk_norm"]),
            ("rms_norm_eps_mode", SUPPORTED["rms_norm_eps_mode"]),
            ("rope_style", SUPPORTED["rope_style"]),
            ("rope_cos_sin_dtype", SUPPORTED["rope_cos_sin_dtype"]),
            ("mudd_accumulation", SUPPORTED["mudd_accumulation"]),
            ("ve_head_slice", SUPPORTED["ve_head_slice"]),
            ("lm_head_precision_mode", SUPPORTED["lm_head_precision_mode"]),
        ):
            value = getattr(self, field)
            if value not in allowed:
                raise NotImplementedError(f"Limite does not implement {field}={value!r}; supported: {sorted(allowed)}")

        if self.rms_norm_has_weight:
            raise NotImplementedError("Limite RMS norm carries no learnable gain (rms_norm_has_weight must be False).")
        if self.sliding_window_convention != SLIDING_WINDOW_CONVENTION:
            raise NotImplementedError(
                f"Unexpected sliding_window_convention {self.sliding_window_convention!r}. "
                "The window span is an off-by-one trap; refusing to guess."
            )
        if self.softcap_logits.get("kind") not in SUPPORTED["softcap_kind"]:
            raise NotImplementedError(f"Limite implements only sigmoid logit softcapping, got {self.softcap_logits!r}")
        if self.final_softcap:
            raise NotImplementedError("Limite does not implement the pre-head tanh cap (final_softcap must be 0).")
        if self.attn_gate_channels < 0:
            raise ValueError("attn_gate_channels must be non-negative.")
        if self.attn_gate_channels:
            if self.attn_gate_scale != 2.0:
                raise NotImplementedError(
                    f"Limite implements the attention gate only as 2 * sigmoid(...), got {self.attn_gate_scale}."
                )
            if self.attn_gate_applied != "per_head_before_o_proj":
                raise NotImplementedError(f"Limite does not implement attn_gate_applied={self.attn_gate_applied!r}.")
        if not self.global_nope:
            raise NotImplementedError("Limite implements rotary-free global layers only (global_nope must be True).")
        if self.rope_per_layer and self.rope_base_local != self.rope_base_global:
            raise NotImplementedError(
                "rope_per_layer with distinct bases is unreachable while global layers skip rotary entirely."
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"num_attention_heads * head_dim ({self.num_attention_heads * self.head_dim}) "
                f"must equal hidden_size ({self.hidden_size})."
            )
        if self.mlp_formula != MLP_FORMULAS[self.mlp_type]:
            raise NotImplementedError(
                f"Limite does not implement mlp_formula={self.mlp_formula!r} for {self.mlp_type!r}."
            )
        if self.mlp_type == "relu2" and self.mlp_ratio * self.hidden_size != self.intermediate_size:
            raise ValueError(
                f"mlp_ratio * hidden_size ({self.mlp_ratio * self.hidden_size}) "
                f"must equal intermediate_size ({self.intermediate_size})."
            )
        if self.ve_dim > self.head_dim:
            raise ValueError(f"ve_dim ({self.ve_dim}) must not exceed head_dim ({self.head_dim}).")
        if self.ve_gate_channels > self.hidden_size:
            raise ValueError(f"ve_gate_channels ({self.ve_gate_channels}) must not exceed hidden_size.")
        if self.ve_stored_heads not in {self.num_attention_heads, self.num_key_value_heads}:
            raise ValueError(
                f"ve_stored_heads ({self.ve_stored_heads}) must be query-head width "
                f"({self.num_attention_heads}) or key-value-head width ({self.num_key_value_heads})."
            )
        if self.padded_vocab_size != self.vocab_size:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) is the width of the embedding and head matrices and must "
                f"equal padded_vocab_size ({self.padded_vocab_size})."
            )
        for name, layers in (
            ("global_layers", self.global_layers),
            ("ve_layers", self.ve_layers),
            ("xsa_layers", self.xsa_layers),
            ("mudd_layers", self.mudd_layers),
        ):
            if layers and (layers[0] < 0 or layers[-1] >= self.num_hidden_layers):
                raise ValueError(f"{name}={layers} is out of range for num_hidden_layers={self.num_hidden_layers}.")

        if self.mudd:
            if sorted(int(k) for k in self.mudd_tap_idx) != list(self.mudd_layers):
                raise ValueError(
                    f"mudd_tap_idx keys {sorted(self.mudd_tap_idx)} must match mudd_layers {self.mudd_layers}. "
                    "The tap topology is inferred upstream and pinned here; a mismatch is silent."
                )
            for layer, taps in self.mudd_tap_idx.items():
                if len(taps) != self.mudd_taps:
                    raise ValueError(f"mudd_tap_idx[{layer}] has {len(taps)} taps, expected {self.mudd_taps}.")
                if max(taps) > int(layer):
                    raise ValueError(
                        f"mudd_tap_idx[{layer}]={taps} reads a history entry that does not exist yet at layer {layer}."
                    )
        elif self.mudd_layers:
            raise ValueError("mudd is disabled but mudd_layers is non-empty.")
        if self.mudd_mlp:
            if not self.mudd:
                raise ValueError("mudd_mlp requires the shared MUDD H-way mixer.")
            if self.mudd_r_site != "resid":
                raise NotImplementedError(
                    f"Limite implements MUDD R only at the residual base, got {self.mudd_r_site!r}."
                )

        if not self.xsa and self.xsa_layers:
            raise ValueError("xsa is disabled but xsa_layers is non-empty.")


__all__ = [
    "LimiteConfig",
    "MLP_FORMULAS",
    "REQUIRED_CONFIG_FIELDS",
    "SLIDING_WINDOW_CONVENTION",
]
