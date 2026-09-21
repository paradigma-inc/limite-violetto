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

from importlib.metadata import PackageNotFoundError, version as package_version

from limite_vllm.contract import ARCHITECTURE, MODEL_TYPE

TARGET_VLLM_VERSION = "0.26.0"


def _validate_vllm_version() -> None:
    """Fail clearly before importing version-specific vLLM internals."""
    try:
        installed = package_version("vllm")
    except PackageNotFoundError:
        # Importing vLLM below will provide the standard missing-package error.
        return

    base_version = installed.split("+", 1)[0]
    if base_version != TARGET_VLLM_VERSION:
        raise RuntimeError(
            f"limite-vllm 0.1.0 supports vLLM {TARGET_VLLM_VERSION}; "
            f"found {installed}. Install the plugin in a compatible vLLM environment."
        )


def register() -> None:
    _validate_vllm_version()

    from transformers import AutoConfig
    from vllm import ModelRegistry

    from limite_vllm.configuration import LimiteConfig
    from limite_vllm.modeling import LimiteForCausalLM

    AutoConfig.register(MODEL_TYPE, LimiteConfig, exist_ok=True)
    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCHITECTURE, LimiteForCausalLM)
