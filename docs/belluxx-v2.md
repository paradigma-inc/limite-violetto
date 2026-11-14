# Belluxx drafting with the native V2 runner

This experimental adapter runs the released
[Belluxx drafter](https://huggingface.co/Belluxx/limite-1b-violetto-drafter)
with Limite Violetto on CUDA. Both workload profiles use the same implementation:

| Profile | Requests in flight | Proposals per target verification |
| --- | ---: | ---: |
| `single-stream` | 1 | 5 |
| `multistream` | up to 32 | 3 |

These are explicit configurations, not automatic concurrency switching.
The combined implementation needs its own completed GPU validation and matched
measurements before release. Results from the earlier separate prototypes do not
establish this implementation's speed or quality.

## Setup and serving

Use the repository's locked vLLM 0.28.0 environment. The adapter checks the exact
source hashes of the vLLM modules it extends. It requires TP=PP=1, asynchronous
scheduling, disabled prefix caching, greedy draft proposals, and vLLM's standard
rejection sampler. Target admission checks the 48-layer, 1280-wide Limite
architecture and BF16 dtype before loading the drafter. The released drafter
remains FP32.

Download the pinned drafter and its reference source. `HF_TOKEN` must be set in
the current environment; these commands never print it:

```bash
hf download Belluxx/limite-1b-violetto-drafter \
  --revision 9f79b9d5e5e0d5d8f48d6d2cfefb4b2fb8926730 \
  --local-dir ../belluxx-drafter --token "$HF_TOKEN"
git clone https://github.com/Belluxx/silicon-violetto.git ../silicon-violetto
git -C ../silicon-violetto checkout 7a3b61aefee583abbc0fb03bd8e4cca2e85492b1
export LIMITE_BELLUXX_WEIGHTS="$PWD/../belluxx-drafter"
export LIMITE_BELLUXX_SOURCE="$PWD/../silicon-violetto"
export VLLM_PLUGINS=limite
export VLLM_USE_V2_MODEL_RUNNER=1
```

The reference loader verifies `tools/spec/drafter.py` before importing it. The
adapter's reference-forward formulas are adapted from that MIT-licensed source;
the attribution and permission notice are in [`licenses/belluxx-MIT.txt`](../licenses/belluxx-MIT.txt).
The drafter checkpoint retains its upstream terms.

Single stream:

```bash
uv run --locked vllm serve paradigma-inc/limite-1b-violetto \
  --revision 19569666aca3b449a4bf0aa0559fa5857681bf54 \
  --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --speculative-config "$(cat configs/belluxx/single-stream.json)"
```

For multistream, use `--max-num-seqs 32` and
`configs/belluxx/multistream.json`. The Python API exposes the same dictionaries
through `limite_vllm.belluxx_config.speculative_config(profile)`.
For the matched baseline, unset `LIMITE_BELLUXX_WEIGHTS`, keep the V2 runner and
all other engine settings, and omit `--speculative-config`.

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
