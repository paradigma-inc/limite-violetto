"""vLLM plugin entry point for the Limite architecture.

Fires in EVERY vLLM process (`VLLM_WORKER_MULTIPROC_METHOD=spawn` re-imports
plugins per worker), so it must be idempotent.

The modeling module is imported eagerly so an invalid runtime cannot appear
registered and then fail only when the first model is loaded.

Registering the transformers config is not optional either: vLLM parses
`config.json` through `AutoConfig` before it ever consults the architecture
string, so without it a Limite checkpoint fails at `ModelConfig` construction
with a misleading "Transformers does not recognize this architecture".
"""

from limite_vllm.contract import ARCHITECTURE, MODEL_TYPE


def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from limite_vllm.configuration import LimiteConfig
    from limite_vllm.modeling import LimiteForCausalLM

    AutoConfig.register(MODEL_TYPE, LimiteConfig, exist_ok=True)
    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCHITECTURE, LimiteForCausalLM)
