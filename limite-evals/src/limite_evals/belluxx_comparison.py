"""Verify and compare complete, possibly sharded, AIME review runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from limite_evals.belluxx_benchmark import digest, info, write


def load_group(root, names):
    manifests, receipts, rows, stats = [], [], [], []
    for name in names:
        run = root / "runs" / name
        m = json.loads((run / "manifest.json").read_text())
        r = json.loads((run / "completion.json").read_text())
        if (run / "runtime-stats.json").exists():
            r["runtime_stats"] = json.loads((run / "runtime-stats.json").read_text())
            if m["model_runner"] == "V2":
                assert all(
                    s["runner_class"] == "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                    for s in r["runtime_stats"]
                )
        assert r["complete"] and r["samples_sha256"] == digest(run / "samples.jsonl")
        samples = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
        keys = [(s["problem_index"], s["rollout_index"]) for s in samples]
        assert len(keys) == len(set(keys)) == r["samples"] == m["expected_samples"]
        assert set(keys) == {tuple(key) for key in m["sample_keys"]}
        for s in samples:
            assert s["completion_tokens"] == len(s["token_ids"]) <= s["max_tokens"]
            assert s["max_tokens"] + s["prompt_tokens"] == m["max_model_len"]
            assert s["protocols"]["exact"]["correct"] is not None
            assert not (s.get("engine_metrics") or {}).get("is_corrupted", False)
        if m["draft_tokens"]:
            draft_stats = json.loads((run / "draft-stats-final.json").read_text())
            if m["model_runner"] == "V2":
                assert all(s["continuity_errors"] == 0 for s in draft_stats)
                assert all(
                    s["implementation"] == "limite_vllm.belluxx_v2.BelluxxSpeculator" for s in draft_stats
                )
            stats.extend(draft_stats)
        manifests.append(m)
        receipts.append(r)
        rows.extend(samples)
    first = manifests[0]
    expected = {(i, j) for i in range(first["problems"]) for j in range(first["rollouts"])}
    observed = [(s["problem_index"], s["rollout_index"]) for s in rows]
    assert len(observed) == len(set(observed)) and set(observed) == expected
    for m in manifests:
        for key in first.keys() - {
            "host",
            "slurm_job_id",
            "shard_index",
            "sample_keys",
            "expected_samples",
            "driver",
        }:
            assert m[key] == first[key], (key, m[key], first[key])
    return manifests, receipts, sorted(rows, key=lambda s: (s["problem_index"], s["rollout_index"])), stats


def summarize(group, bootstrap):
    import numpy as np

    manifests, receipts, rows, stats = group
    m = manifests[0]
    n, k = m["problems"], m["rollouts"]
    correct = np.array([s["protocols"]["exact"]["correct"] for s in rows]).reshape(n, k)
    lengths = np.array([s["completion_tokens"] for s in rows]).reshape(n, k)
    latency = np.array([s["latency_s"] for s in rows]).reshape(n, k)
    elapsed = sum(r["elapsed_s"] for r in receipts)
    metrics = {
        "problems": n,
        "rollouts": k,
        "samples": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accuracy_problem_bootstrap_95ci": np.quantile(
            correct.mean(1)[bootstrap].mean(1), [0.025, 0.975]
        ).tolist(),
        "solved_at_least_once": int(correct.any(1).sum()),
        "truncated": sum(s["finish_reason"] == "length" for s in rows),
        "tokens": int(lengths.sum()),
        "summed_gpu_evaluation_seconds": elapsed,
        "tokens_per_gpu_second": float(lengths.sum() / elapsed),
        "median_tokens": float(np.median(lengths)),
        "median_latency_s": float(np.median(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "median_request_tokens_per_s": float(np.median(lengths / latency)),
        "concurrency_per_gpu": m["concurrency"],
        "gpu_replicas": len(manifests),
        "source_commit": m["git_commit"],
    }
    runtime_stats = [s for r in receipts for s in r.get("runtime_stats", [])]
    metrics["peak_gpu_allocated_bytes"] = max(
        (s["peak_allocated_bytes"] for s in runtime_stats), default=None
    )
    metrics["gpu_uuids"] = sorted({s["gpu_uuid"] for s in runtime_stats})
    metrics["memory_scope"] = "Peak torch allocation including the preallocated vLLM KV pool."
    ttft = [(s.get("engine_metrics") or {}).get("first_token_latency", 0) for s in rows]
    metrics["median_first_token_latency_s"] = float(np.median(ttft)) if all(ttft) else None
    if stats:
        totals = {
            key: sum(s[key] for s in stats)
            for key in ("calls", "proposed", "accepted", "target_steps", "emitted", "context_checks")
        }
        totals["acceptance_rate"] = totals["accepted"] / totals["proposed"]
        totals["emitted_tokens_per_target_step"] = totals["emitted"] / totals["target_steps"]
        buckets = {}
        for s in stats:
            for label, values in s.get("by_context", {}).items():
                bucket = buckets.setdefault(label, {key: 0 for key in values})
                for key, value in values.items():
                    bucket[key] += value
        totals["by_context"] = buckets
        metrics["draft"] = totals
    return metrics, correct, lengths, latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, nargs="+")
    parser.add_argument("--speculative", required=True, nargs="+")
    parser.add_argument("--out-id", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = args.root / "reports" / args.out_id
    if out.exists():
        raise FileExistsError(out)
    for name in args.baseline + args.speculative:
        for filename in ("manifest.json", "completion.json", "samples.jsonl"):
            if not (args.root / "runs" / name / filename).is_file():
                raise FileNotFoundError(args.root / "runs" / name / filename)
    if args.check:
        info("Comparison inputs and output isolation validated")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = load_group(args.root, args.baseline)
    draft = load_group(args.root, args.speculative)
    a, b = baseline[0][0], draft[0][0]
    differences = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    allowed = {
        "drafter_profile",
        "draft_model",
        "draft_revision",
        "draft_tokens",
        "drafter_source",
        "drafter_source_sha256",
        "draft_sampling",
        "draft_dtype",
        "verification",
        "host",
        "slurm_job_id",
        "driver",
    }
    assert differences <= allowed, differences - allowed
    assert a["draft_tokens"] == 0 < b["draft_tokens"]
    assert a["budget_policy"] == "full_context_minus_each_rendered_prompt"
    assert a["workload"] == b["workload"] == "aime_2026"
    assert a["problems"] == 30 and a["rollouts"] == 4
    rng = np.random.default_rng(20261112)
    bootstrap = rng.integers(0, a["problems"], size=(10000, a["problems"]))
    base, ac, al, at = summarize(baseline, bootstrap)
    spec, bc, bl, bt = summarize(draft, bootstrap)
    if not base["gpu_uuids"] or base["gpu_uuids"] != spec["gpu_uuids"]:
        raise ValueError("Matched runs must use the same physical GPU set")
    delta = bc.mean(1) - ac.mean(1)
    speed = (bl.sum(1)[bootstrap].sum(1) / bt.sum(1)[bootstrap].sum(1)) / (
        al.sum(1)[bootstrap].sum(1) / at.sum(1)[bootstrap].sum(1)
    )
    comparison = {
        "coverage_and_contract_verified": True,
        "baseline_runs": args.baseline,
        "speculative_runs": args.speculative,
        "baseline": base,
        "speculative": spec,
        "accuracy_delta_percentage_points": float(delta.mean() * 100),
        "accuracy_delta_problem_bootstrap_95ci_pp": (
            np.quantile(delta[bootstrap].mean(1), [0.025, 0.975]) * 100
        ).tolist(),
        "aggregate_throughput_ratio": spec["tokens_per_gpu_second"] / base["tokens_per_gpu_second"],
        "request_time_normalized_token_rate_ratio_95ci": np.quantile(speed, [0.025, 0.975]).tolist(),
        "interval_scope": "Whole-problem bootstrap of tokens / summed request latency. This is per-request rate, not a repeated-run interval for wall-clock aggregate throughput.",
        "timing_scope": "Completed output tokens / summed per-GPU evaluation wall time. Initialization and explicit warmup excluded; residual compilation, prefill, grading and batch drain included.",
        "draft_counter_scope": "Verified proposals and kernel-emitted tokens before final EOS/length clipping; may include final overshoot.",
        "native_metal_or_int8_measured": False,
        "comparison_kind": "matched V2 drafter comparison",
    }
    out.mkdir(parents=True)
    write(out / "summary.json", comparison)
    with (out / "problem_metrics.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "problem",
                "baseline_correct",
                "draft_correct",
                "baseline_tokens",
                "draft_tokens",
                "baseline_latency_s",
                "draft_latency_s",
            ]
        )
        for i in range(a["problems"]):
            writer.writerow([i, ac[i].sum(), bc[i].sum(), al[i].sum(), bl[i].sum(), at[i].sum(), bt[i].sum()])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    colors = ["#335C67", "#E09F3E"]
    axes[0].bar(["Baseline", "+ Belluxx"], [base["accuracy"] * 100, spec["accuracy"] * 100], color=colors)
    axes[0].set(ylabel="AIME 2026 accuracy (%)", ylim=(0, 100))
    axes[1].bar(
        ["Baseline", "+ Belluxx"],
        [base["tokens_per_gpu_second"], spec["tokens_per_gpu_second"]],
        color=colors,
    )
    axes[1].set(ylabel="Output tokens per GPU-second")
    figure.suptitle(f"H100 CUDA comparison · {a['concurrency']} stream(s)/GPU · 30 problems × 4 rollouts")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=180, facecolor="white")
    plt.close(figure)
    report = f"""# Limite Violetto with Belluxx drafter: CUDA comparison

All 30 AIME 2026 problems completed with four rollouts per problem in each arm.
Both arms use one H100 per replica, BF16 target, canonical prompts,
temperature 0.6/top-p 0.95, identical request seeds and full remaining context.
Runtime comparison: {comparison["comparison_kind"]}. Baseline runner {a["model_runner"]},
candidate runner {b["model_runner"]}; cross-runtime results include engine differences.
The candidate adds the released FP32 drafter with {b["draft_tokens"]} deterministic proposals,
verified using vLLM's rejection sampler. This is a CUDA chain integration;
Apple Metal kernels, the author's tree strategy, and int8 execution are unmeasured.

| Metric | Baseline | Baseline + Belluxx |
|---|---:|---:|
| Correct / 120 | {base["correct"]} | {spec["correct"]} |
| Mean accuracy | {base["accuracy"]:.2%} | {spec["accuracy"]:.2%} |
| Problems solved at least once / 30 | {base["solved_at_least_once"]} | {spec["solved_at_least_once"]} |
| Truncated samples | {base["truncated"]} | {spec["truncated"]} |
| Output tokens | {base["tokens"]:,} | {spec["tokens"]:,} |
| Summed evaluation time / GPU, seconds | {base["summed_gpu_evaluation_seconds"]:.2f} | {spec["summed_gpu_evaluation_seconds"]:.2f} |
| Tokens per GPU-second | {base["tokens_per_gpu_second"]:.2f} | {spec["tokens_per_gpu_second"]:.2f} |
| Median request latency, seconds | {base["median_latency_s"]:.2f} | {spec["median_latency_s"]:.2f} |
| p95 request latency, seconds | {base["p95_latency_s"]:.2f} | {spec["p95_latency_s"]:.2f} |

Accuracy difference: {comparison["accuracy_delta_percentage_points"]:+.2f} percentage points;
whole-problem bootstrap 95% interval {comparison["accuracy_delta_problem_bootstrap_95ci_pp"]}.
Throughput ratio (candidate/baseline): {comparison["aggregate_throughput_ratio"]:.3f}×.
The bootstrap resamples 30 problems, retaining all four rollouts in each problem.
It measures uncertainty across this small benchmark, not equivalence or a formal
non-inferiority guarantee. Identical seeds do not imply identical stochastic token
paths under speculative decoding.

{comparison["timing_scope"]}
For sharded single-stream runs, summed GPU time is used; parallel replicas are not
mistaken for a single-GPU throughput improvement. Counter scope:
{comparison["draft_counter_scope"]}

Draft acceptance: {spec["draft"]["acceptance_rate"]:.2%}; emitted tokens per target
step: {spec["draft"]["emitted_tokens_per_target_step"]:.3f}.
The checkpoint is not a replacement target model: verified draft tokens still
follow the target sampler. Sampling variation and numerical execution can change
individual answers; reduced target steps alone do not prove faster execution.

All expected request keys, full-context budgets, token lengths, scoring coverage,
completion hashes, and matching measurement manifests were verified. Per-problem
metrics, source commits, acceptance counters, and machine-readable results are in
the adjacent CSV/JSON. Raw generations and manifests remain in the named run directories.
"""
    (out / "report.md").write_text(report)
    info(f"Verified comparison written to {out}")


if __name__ == "__main__":
    main()
