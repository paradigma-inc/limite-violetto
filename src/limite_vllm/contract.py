"""Stable architecture and token identities consumed by the vLLM plugin."""

MODEL_TYPE = "limite"
ARCHITECTURE = "LimiteForCausalLM"

BOS_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643

__all__ = [
    "ARCHITECTURE",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "MODEL_TYPE",
    "PAD_TOKEN_ID",
]
