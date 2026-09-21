# Limite vLLM

Optimized out-of-tree vLLM implementation of the Limite architecture. This
repository contains serving code only: model weights, tokenizer assets, and the
Transformers implementation are distributed separately through the model
repository.

## Compatibility

The current compatibility target is vLLM `0.26.0`. Production environments
should pin that exact version until CI covers a wider range. vLLM is
deliberately not a package dependency because deployment images own CUDA,
PyTorch, and vLLM as one tested stack.

The plugin is pure Python and requires Python 3.12. It uses the Transformers
configuration API already present in a vLLM environment.

## Installation

Install a tagged release directly from GitHub:

```bash
python -m pip install \
  "limite-vllm @ git+https://github.com/paradigma-inc/limite-violetto.git@v0.1.0"
```

For local development:

```bash
uv sync --group dev
```

The package registers `LimiteForCausalLM` through the
`vllm.general_plugins` entry-point group under the plugin name `limite`.

## Serving

After installing the plugin into an environment that provides a compatible
vLLM build:

```bash
VLLM_PLUGINS=limite vllm serve <organization>/<model>
```

The model repository must provide the weights and tokenizer, and its
`config.json` must contain:

```json
{
  "model_type": "limite",
  "architectures": ["LimiteForCausalLM"]
}
```

All architecture-specific configuration fields remain mandatory and are
validated before the vLLM graph is constructed. The plugin registers its own
minimal `LimiteConfig`, so serving does not depend on another source checkout
or on loading the Transformers model implementation.

## Build and checks

```bash
uv run pytest
uv run ruff check .
uv build
```

The generated wheel contains only the `limite_vllm` Python package. It does not
contain weights, tokenizer assets, Hub export code, or a Transformers execution
graph.
