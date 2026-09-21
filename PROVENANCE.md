# Limite vLLM plugin extraction

This standalone distribution was extracted from Paradigma `staging` at commit
`51dc058457fa53315183fb58ae17b5bbc4f12206`.

## Canonical ownership

This repository owns only the optimized vLLM plugin. Its small
`PretrainedConfig` subclass exists solely so vLLM can parse and validate the
checkpoint configuration without importing code from another checkout. The
Transformers implementation and model weights are distributed separately.

`limite_vllm` owns the vLLM configuration contract, validation, registration,
weight loading, and single optimized execution graph.

## Extraction changes

- The model and Python package identities were cut over to Limite.
- The alternate execution graph and runtime selector were removed.
- The fused QKV and SwiGLU graph became the single execution graph.
- Projection scalars are folded exactly once after every exhaustive load.
- The native Transformers graph and Hugging Face export tooling are excluded.
- Standalone documentation and contract tests replace monorepo integration
  metadata.
