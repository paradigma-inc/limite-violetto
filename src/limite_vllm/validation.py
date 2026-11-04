"""vLLM-specific validation layered on the Limite checkpoint contract."""

from __future__ import annotations

from limite_vllm.configuration import LimiteConfig

#: The attention shape the Triton prologue and epilogue hard-code: the released
#: Limite 1B checkpoints. XSA must also cover every layer.
KERNEL_ATTENTION_SHAPE: dict[str, object] = {
    "hidden_size": 1280,
    "num_attention_heads": 10,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "ve_dim": 128,
    "ve_stored_heads": 2,
    "ve_gate_channels": 12,
    "ve_gate_scale": 2.0,
    "attn_gate_channels": 128,
    "xsa": True,
    "xsa_normalize_eps": 1e-4,
}


def validate(config: LimiteConfig) -> None:
    """Reject checkpoint features the Limite graph does not implement."""
    config.validate_architecture()
    if getattr(config, "global_window", None) != -1:
        raise NotImplementedError(
            "Limite vLLM implements global layers only with global_window=-1."
        )
    if getattr(config, "tie_word_embeddings", None) is not True:
        raise NotImplementedError(
            "Limite vLLM requires tied embedding and LM-head weights."
        )
    differs = {
        field: getattr(config, field, None)
        for field, required in KERNEL_ATTENTION_SHAPE.items()
        if getattr(config, field, None) != required
    }
    xsa_layers = sorted(int(layer) for layer in config.xsa_layers)
    if xsa_layers != list(range(int(config.num_hidden_layers))):
        differs["xsa_layers"] = xsa_layers
    if differs:
        raise NotImplementedError(
            "Limite vLLM's attention kernels implement the released Limite 1B "
            f"attention shape {KERNEL_ATTENTION_SHAPE} with XSA on every layer; "
            f"this checkpoint differs in {differs}."
        )
