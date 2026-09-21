"""vLLM-specific validation layered on the Limite checkpoint contract."""

from __future__ import annotations

from limite_vllm.configuration import LimiteConfig


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
