# Belluxx drafting with the native V2 runner

This experimental adapter runs the released
[Belluxx drafter](https://huggingface.co/Belluxx/limite-1b-violetto-drafter)
with Limite Violetto on CUDA. Both workload profiles use the same implementation:

| Profile | Requests in flight | Proposals per target verification |
| --- | ---: | ---: |
| `single-stream` | 1 | 5 |
| `multistream` | up to 32 | 3 |

These are explicit configurations, not automatic concurrency switching.
The profile helper sets only the proposal count; set the request ceiling separately
as shown below. Performance and quality measurements apply to their exact combined
source revision and workload. Results from the earlier separate prototypes do not
establish this implementation's speed or quality; throughput gains alone do not
establish accuracy equivalence.

## Setup and serving

Use Linux x86-64, Python 3.12, an NVIDIA GPU and a CUDA 13.0-compatible driver,
as in the [root quickstart](../README.md#quickstart). Allocate the GPU before
model operations. From a checkout containing this adapter, install the locked
environment without activating a shell environment:

```bash
uv sync --locked
```

Use the repository's locked vLLM 0.28.0 environment. The adapter checks the exact
source hashes of the vLLM modules it extends. It requires TP=PP=1, asynchronous
scheduling, disabled prefix caching and KV-transfer connectors, greedy draft
proposals, and vLLM's standard rejection sampler. External KV caches cannot
restore the drafter's accepted feature history, even when local prefix caching
is disabled. Target admission checks the 48-layer, 1280-wide Limite
architecture and BF16 dtype before loading the drafter. The released drafter
remains FP32.

Download the pinned drafter and its reference source. `HF_TOKEN` must be set in
the current environment; these commands never print it:

```bash
: "${HF_TOKEN:?Export HF_TOKEN before downloading weights}"
uv run --locked hf download Belluxx/limite-1b-violetto-drafter \
  --revision 9f79b9d5e5e0d5d8f48d6d2cfefb4b2fb8926730 \
  --local-dir ../belluxx-drafter --token "$HF_TOKEN"
git clone https://github.com/Belluxx/silicon-violetto.git ../silicon-violetto
git -C ../silicon-violetto checkout 7a3b61aefee583abbc0fb03bd8e4cca2e85492b1
export LIMITE_BELLUXX_WEIGHTS="$PWD/../belluxx-drafter"
export LIMITE_BELLUXX_SOURCE="$PWD/../silicon-violetto"
export VLLM_PLUGINS=limite
export VLLM_USE_V2_MODEL_RUNNER=1
```

Set these exports before initializing vLLM, including in worker processes.
Select only the `limite` plugin in a process using this adapter. Other adapters
that patch the same vLLM hooks cannot be composed in the same process; use a
fresh process when changing plugin selection. The weights directory must contain
`model.safetensors`; the source directory must contain `tools/spec/drafter.py`.

The reference loader verifies Belluxx's original `tools/spec/drafter.py` before
importing it. Paradigma developed the CUDA/vLLM integration, adaptations, and
optimizations under Apache-2.0. The original MIT copyright and permission notice
is preserved for incorporated upstream material in
[`licenses/belluxx-MIT.txt`](../licenses/belluxx-MIT.txt).
The drafter checkpoint retains its upstream terms.

Single stream:

```bash
uv run --locked vllm serve paradigma-inc/limite-1b-violetto \
  --revision 19569666aca3b449a4bf0aa0559fa5857681bf54 \
  --hf-token "$HF_TOKEN" --dtype bfloat16 \
  --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --speculative-config "$(cat configs/belluxx/single-stream.json)"
```

For multistream, use `--max-num-seqs 32` and
`configs/belluxx/multistream.json`. The target is downloaded at the pinned revision
on first launch. Send normal chat requests to `http://localhost:8000/v1/chat/completions`
using the [serving example](../README.md#serving); the server applies the checkpoint's
chat template. Set temperature 0.6 and top-p 0.95 explicitly if your client supplies
its own sampling defaults, and allow the remaining context for generation.

## Python generation

Use the same environment and exports. Save this as a Python file and run it with
`uv run --locked python <file.py>`; the main guard supports worker spawning.
Pass the already-rendered token IDs so the native template and special tokens
are applied exactly once:

```python
import os

from vllm import LLM, SamplingParams

from limite_vllm.belluxx_config import speculative_config


def main():
    context_length = 131072
    llm = LLM(
        model="paradigma-inc/limite-1b-violetto",
        revision="19569666aca3b449a4bf0aa0559fa5857681bf54",
        hf_token=os.environ["HF_TOKEN"],
        dtype="bfloat16",
        max_model_len=context_length,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        speculative_config=speculative_config("single-stream"),
    )
    ids = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": "Solve 2x + 3 = 11."}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    remaining = context_length - len(ids)
    if remaining <= 0:
        raise ValueError("Prompt leaves no room for generation")
    result = llm.generate(
        [{"prompt_token_ids": ids}],
        SamplingParams(temperature=0.6, top_p=0.95, max_tokens=remaining),
    )
    print(result[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

For multistream, select `speculative_config("multistream")` and `max_num_seqs=32`.
The helper does not configure concurrency, prefix caching, or environment variables.
It ships in the plugin wheel; the JSON presets and this guide belong to the source
checkout. Installing that wheel in another project does not inherit this repository's
uv CUDA sources or lockfile; see [plugin-only installation](../README.md#install-only-the-plugin).

## Disable drafting and inspect startup

To turn drafting off, start a fresh process with `LIMITE_BELLUXX_WEIGHTS` unset and
omit `speculative_config` / `--speculative-config`. Keep V2 and the other engine
settings for a matched baseline. Unsetting the variable inside an already-running
Python process does not undo its installed hooks.

| Startup error | Action |
| --- | --- |
| Missing `LIMITE_BELLUXX_SOURCE`, weights, or reference file | Set both asset paths before startup and check the files named above. |
| Upstream source hash or async compatibility check changed | Restore the locked runtime and pinned reference source; another version requires porting and validation. Keep the checks enabled. |
| Configuration hooks cannot be inspected | Start a fresh process with `VLLM_PLUGINS=limite` and no other adapter patching vLLM. |
| Unsupported target, dtype, parallelism, scheduling, or cache | Use the pinned target, BF16, TP=PP=1, async scheduling, and disable prefix caching and KV transfer. |

For Python diagnostics, add
`worker_extension_cls="limite_vllm.belluxx_v2.BelluxxV2WorkerExtension"` to the
`LLM` constructor. Inside `main`, after warmup and after measured generation
respectively, call:

```python
llm.collective_rpc("reset_drafter_stats")  # after warmup, before measurement
# Run the measured requests here.
stats = llm.collective_rpc("finish_drafter")
identity = llm.collective_rpc("finish_runtime")
print(stats, identity)
```

Each result contains one dictionary per worker. Optionally set
`LIMITE_BELLUXX_STATS` to a new writable file path before startup; its parent
directory must already exist. That file is written by `finish_drafter`, not
automatically when a CLI server exits. The benchmark runner performs these
diagnostic calls itself.

## Shared cache and verification

The target emits auxiliary states after layers 35 and 47. The drafter normalizes
them in FP32, rounds the features to BF16 as in its training path, and fuses them
in FP32. Each new token's K/V projections are computed once. Only the accepted
prefix enters a 1,024-position FP32 ring indexed by persistent request slots.
Rejected tokens never enter retained history. Recomputed requests restart at
position zero; slot reuse and long prefill chunks are covered by tests.

Device-side continuity checks avoid a CPU synchronization on every decode step.
Padding uses common power-of-two batch sizes; gathering uses the runtime request
count so batch drain does not create one kernel specialization per active count.
The draft forward uses `torch.compile`; the target retains ownership of its CUDA
graphs. vLLM's native scheduler clips proposals at the context limit, and its
standard rejection sampler owns verification and target KV rollback.

The worker extension records the actual runner, proposer, verifier module, GPU
UUID, acceptance counters, and history continuity. Peak allocator readings include
vLLM's preallocated KV pool and are not the drafter's incremental memory cost.
Using the standard verifier does not establish bitwise equality across different
execution shapes; numerical and quality comparisons remain empirical checks.

## Reproduce the comparison

Allocate a suitable GPU before running model or dataset operations. The evaluator
does not select a scheduler or allocate hardware. From `limite-evals/`, use its
locked environment and prepare immutable inputs into a new directory:

```bash
uv sync --locked
uv run --no-sync python -m limite_evals.belluxx_benchmark prepare \
  --root ../outputs/belluxx-inputs
```

The preparation pins both model revisions above and `MathArena/aime_2026` revision
`10b4e45b7a503075d4da8a0d57916a4f06ce6bd2`, records the native chat template and token
IDs, and gives each problem the full remaining capacity of the 131,072-token context.
Use a fresh output root for each comparison. Run baseline and candidate on the same
physical GPU, then repeat with the order reversed:

```bash
uv run --no-sync python -m limite_evals.belluxx_benchmark run \
  --root ../outputs/belluxx-inputs --output-root ../outputs/comparison/runs \
  --run-id baseline-c1 --concurrency 1 --rollouts 4
uv run --no-sync python -m limite_evals.belluxx_benchmark run \
  --root ../outputs/belluxx-inputs --output-root ../outputs/comparison/runs \
  --run-id draft-c1 --concurrency 1 --rollouts 4 \
  --drafter-profile single-stream --drafter-source "$LIMITE_BELLUXX_SOURCE"
uv run --no-sync python -m limite_evals.belluxx_comparison \
  --root ../outputs/comparison --baseline baseline-c1 --speculative draft-c1 \
  --out-id single-stream
```

Repeat with concurrency 32, distinct run IDs, and `--drafter-profile multistream`.
An optional `limite_evals.belluxx_sequence --config sequence.json` accepts a JSON
list of benchmark argument lists and retains subprocess timing and exit receipts.
All three tools expose `--check` to validate arguments and paths without loading
models or datasets. Each run must use the same committed source; the manifest
records the commit, runtime source hashes, input identity, and sampling settings.

The protocol uses temperature 0.6, top-p 0.95, seeds 0–119 and four rollouts per
problem. It retains every truncation. Both arms warm the same batch shapes before
timing. Output tokens/GPU-second includes request admission, prefill, generation,
residual compilation, scoring and drain; startup and explicit warmup are recorded
separately. Reports include complete sample/hash checks, throughput, latency,
acceptance and a whole-problem bootstrap interval for the accuracy difference.
Keep repeated runs separate. A request-time bootstrap is not an uncertainty
interval for repeated aggregate throughput, and a non-significant accuracy
difference does not establish equivalence or non-inferiority.

Use `--diagnostic --max-new-tokens N` for bounded **non-AIME** diagnostics.
Long repeated prompts, lower context limits, native EOS, and token log probabilities
are available for boundary and numerical checks. When lowering the context limit,
set `--max-batched-tokens` no higher than that limit. These runs cannot enter the AIME
quality comparison. Do not tune configurations using AIME answers.

## Validation

Run the package and evaluator tests in the locked environment. For GPU contract
tests, set `LIMITE_BELLUXX_SOURCE`, `LIMITE_BELLUXX_WEIGHTS`,
`LIMITE_TARGET_WEIGHTS` to the pinned local target snapshot, and
`RUN_BELLUXX_V2_ENGINE_TEST=1`, with `VLLM_USE_V2_MODEL_RUNNER=1`.
`tests/test_belluxx_v2.py` checks reference parity at 1/3/5/7 proposals, accepted
history under rejection/wrap/reordering/reuse, and complete context exhaustion
at both selected proposal counts. Use a fresh process for each actual-engine test.

Primary Flywheel node: unresolved; curated scientific logging is deferred until
the project-to-node mapping is explicit. No graph node is inferred from this document.
