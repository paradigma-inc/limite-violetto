# Scoring protocols

Mathematical completions are evaluated with all available views:

- `exact`: the benchmark's published grader, executed from digest-pinned source;
- `reference`: the verifier used by the evaluation environment;
- `lenient` and `permissive`: documented relaxations of `exact` where a
  published grader is bound, falling back to `reference` where it is not; the
  base is recorded per taskset;
- `format_ok`, truncation, and comparison-timeout diagnostics.

The `exact` and `reference` columns are independent. Neither is assumed to be a
subset of the other. If a benchmark does not publish a compatible grader, the
column is absent with a reason rather than reported as zero.

The published sources are MATH/Hendrycks, OpenAI PRM800K, and MathArena.
Their repository revisions, paths, and hashes are recorded in
`packages/limite-evals-core/src/limite_evals_core/published/vendor/PROVENANCE.md`.
