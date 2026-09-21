# Limite vLLM

Optimized out-of-tree vLLM implementation of the Limite architecture. This
repository contains serving code only: model weights, tokenizer assets, and the
Transformers implementation are distributed separately through the model
repository.

## Requirements

- Python `3.12`
- vLLM `0.26.0`
- A CUDA and PyTorch stack supported by that vLLM installation
- A Limite model repository containing the weights, tokenizer, and configuration

Install vLLM following its platform-specific instructions before installing
this plugin. The plugin deliberately declares no runtime dependencies: it does
not install or upgrade CUDA, PyTorch, vLLM, or Transformers. Those packages are
owned by the serving environment and must be tested as one stack. The plugin
fails at startup with a clear message when the installed vLLM release is not
`0.26.0`; local build suffixes such as `0.26.0+cu129` are accepted.

## Installation

Install a tagged release directly from GitHub:

```bash
python -m pip install \
  "limite-vllm @ git+https://github.com/paradigma-inc/limite-violetto.git@v0.1.0"
```

This installs only the lightweight Limite plugin into the active vLLM
environment. It does not download model weights; vLLM retrieves them from the
model repository when serving starts.

## Models

Hugging Face organization: [`paradigma-inc`](https://huggingface.co/paradigma-inc)

Available model names:

- `limite-1b-base`
- `limite-1b-base-soup`
- `limite-1b-violetto`

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
VLLM_PLUGINS=limite vllm serve paradigma-inc/<model>
```

Replace `<model>` with one of the model names listed above.

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
