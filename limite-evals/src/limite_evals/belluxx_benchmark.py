"""Pinned AIME comparison of the CUDA target with and without Belluxx drafting."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from rich.console import Console

_console = Console(force_terminal=True)

TARGET = "paradigma-inc/limite-1b-violetto"
TARGET_REV = "19569666aca3b449a4bf0aa0559fa5857681bf54"
DRAFTER = "Belluxx/limite-1b-violetto-drafter"
DRAFTER_REV = "9f79b9d5e5e0d5d8f48d6d2cfefb4b2fb8926730"
DATASET = "MathArena/aime_2026"
DATASET_REV = "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def info(message: str) -> None:
    _console.print(f"[cyan]INFO[/cyan] {message}")


def verify_inputs(root: Path, identity: dict) -> None:
    """Verify the pinned prepared inputs before any engine or timing begins."""
    if (identity["dataset"], identity["dataset_revision"]) != (DATASET, DATASET_REV):
        raise ValueError("Prepared dataset differs from the pinned AIME revision")
    if digest(root / "inputs.json") != identity["inputs_sha256"]:
        raise ValueError("Prepared prompt inputs have changed")
    for name, repo, revision in (("target", TARGET, TARGET_REV), ("drafter", DRAFTER, DRAFTER_REV)):
        snapshot = identity["snapshots"][name]
        if (snapshot["repo"], snapshot["revision"]) != (repo, revision):
            raise ValueError(f"Prepared {name} revision differs from the pinned checkpoint")
        for filename, expected in snapshot["sha256"].items():
            if digest(root / "models" / name / filename) != expected:
                raise ValueError(f"Prepared {name}/{filename} has changed")


def prepare(root: Path) -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    token = os.environ["HF_TOKEN"]
    snapshots = {}
    for name, repo, revision in (("target", TARGET, TARGET_REV), ("drafter", DRAFTER, DRAFTER_REV)):
        destination = root / "models" / name
        snapshot_download(
            repo,
            revision=revision,
            token=token,
            local_dir=destination,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.py",
                "*.safetensors",
                "*.vlt",
                "README.md",
                "LICENSE",
            ],
        )
        snapshots[name] = {
            "repo": repo,
            "revision": revision,
            "path": str(destination),
            "sha256": {
                str(p.relative_to(destination)): digest(p) for p in destination.iterdir() if p.is_file()
            },
        }
    tokenizer = AutoTokenizer.from_pretrained(
        root / "models/target", trust_remote_code=True, token=token
    )
    config = json.loads((root / "models/target/config.json").read_text())
    generation = json.loads((root / "models/target/generation_config.json").read_text())
    data = load_dataset(DATASET, revision=DATASET_REV, split="train", token=token)
    assert len(data) == 30, f"Expected 30 AIME problems, got {len(data)}"
    records = []
    for i, row in enumerate(data):
        # The shipped template supplies the canonical math system instruction.
        prompt = row["problem"]
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        assert ids == tokenizer.encode(rendered, add_special_tokens=False)
        records.append(
            {
                "problem_index": i,
                "source": row,
                "gold": str(int(row["answer"])),
                "prompt": prompt,
                "rendered": rendered,
                "prompt_token_ids": ids,
                "max_tokens": config["max_position_embeddings"] - len(ids),
            }
        )
    write(root / "inputs.json", records)
    write(
        root / "identity.json",
        {
            "snapshots": snapshots,
            "dataset": DATASET,
            "dataset_revision": DATASET_REV,
            "inputs_sha256": digest(root / "inputs.json"),
            "config": config,
            "generation_config": generation,
            "tokenizer": {
                "chat_template": tokenizer.chat_template,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "assistant_role": "assistant",
                "add_generation_prompt": True,
                "add_special_tokens": False,
                "continuation_span": "after prompt_token_ids",
            },
        },
    )
    info(f"Prepared {len(records)} problems and both pinned checkpoints in {root}")


def run(args: argparse.Namespace) -> None:
    output = (args.output_root or args.root / "runs") / args.run_id
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    if args.draft_tokens:
        os.environ["LIMITE_BELLUXX_WEIGHTS"] = str(args.root / "models/drafter")
        os.environ["LIMITE_BELLUXX_SOURCE"] = str(args.drafter_source)
        os.environ["LIMITE_BELLUXX_STATS"] = str(output / "draft-stats.json")
    else:
        for name in ("LIMITE_BELLUXX_WEIGHTS", "LIMITE_BELLUXX_SOURCE", "LIMITE_BELLUXX_STATS"):
            os.environ.pop(name, None)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from limite_evals import suites, tasksets
    from limite_evals_core.protocols import score

    root = args.root
    output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((root / "identity.json").read_text())
    verify_inputs(root, identity)
    inputs = json.loads((root / "inputs.json").read_text())
    if args.diagnostic:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(root / "models/target"), trust_remote_code=True, token=os.environ["HF_TOKEN"]
        )
        prompts = [
            "Explain why the sum of the first n odd positive integers is n squared.",
            "Write a clear proof that there are infinitely many prime numbers.",
            "Explain how to solve a quadratic equation by completing the square.",
            "Derive the sum of a finite geometric series and give examples.",
        ]
        inputs = []
        for i in range(args.limit or 4):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i % len(prompts)] * args.diagnostic_repeat}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            inputs.append(
                dict(problem_index=i, prompt_token_ids=tokens, max_tokens=args.max_new_tokens, gold=None)
            )
    elif args.limit:
        inputs = inputs[: args.limit]
    if args.max_new_tokens is not None:
        for record in inputs:
            record["max_tokens"] = min(record["max_tokens"], args.max_new_tokens)
    config = identity["config"]
    if args.diagnostic_context_length is not None:
        if args.diagnostic_context_length > config["max_position_embeddings"]:
            raise ValueError("Diagnostic context exceeds the supported context")
        config = dict(config, max_position_embeddings=args.diagnostic_context_length)
    if args.diagnostic:
        for record in inputs:
            remaining = config["max_position_embeddings"] - len(record["prompt_token_ids"])
            if remaining < 1:
                raise ValueError("Diagnostic prompt leaves no generation capacity")
            record["max_tokens"] = min(record["max_tokens"], remaining)
        write(output / "diagnostic-inputs.json", inputs)
    stop = identity["generation_config"]["eos_token_id"]
    stop = stop if isinstance(stop, list) else [stop]
    pending = [
        (record, rollout)
        for record in inputs
        for rollout in range(args.rollouts)
        if (record["problem_index"] * args.rollouts + rollout) % args.shards == args.shard_index
    ]
    warmup_sizes = sorted({2**i for i in range(args.concurrency.bit_length())} | {args.concurrency})
    contract = {
        "workload": "diagnostic_non_aime" if args.diagnostic else "aime_2026",
        "diagnostic_repeat": args.diagnostic_repeat,
        "diagnostic_logprobs": args.diagnostic_logprobs,
        "ignore_eos": args.diagnostic and not args.diagnostic_stop_at_eos,
        "drafter_profile": args.drafter_profile,
        "model": TARGET,
        "revision": TARGET_REV,
        "dataset": None if args.diagnostic else DATASET,
        "dataset_revision": None if args.diagnostic else DATASET_REV,
        "rollouts": args.rollouts,
        "problems": len(inputs),
        "expected_samples": len(pending),
        "shards": args.shards,
        "shard_index": args.shard_index,
        "sample_keys": [[record["problem_index"], rollout] for record, rollout in pending],
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": -1,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "max_model_len": config["max_position_embeddings"],
        "budget_policy": "full_context_minus_each_rendered_prompt"
        if args.max_new_tokens is None
        else "bounded_runtime_validation",
        "stop_token_ids": stop,
        "dtype": "bfloat16",
        "draft_model": DRAFTER if args.draft_tokens else None,
        "draft_revision": DRAFTER_REV if args.draft_tokens else None,
        "draft_tokens": args.draft_tokens,
        "drafter_source": str(args.drafter_source) if args.draft_tokens else None,
        "drafter_source_sha256": digest(args.drafter_source / "tools/spec/drafter.py")
        if args.draft_tokens
        else None,
        "draft_sampling": "deterministic parallel chain" if args.draft_tokens else None,
        "draft_dtype": "float32" if args.draft_tokens else None,
        "verification": "vLLM standard rejection sampler" if args.draft_tokens else None,
        "runtime": "limite-vllm repository source",
        "model_runner": args.model_runner.upper(),
        "async_scheduling": "auto",
        "cuda_graph_mode": "FULL_AND_PIECEWISE" if not args.eager else "NONE",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
        ),
        "identity_sha256": digest(root / "identity.json"),
        "module_sha256": digest(Path(__file__)),
        "runtime_source_sha256": {
            path.name: digest(path)
            for path in sorted((Path(__file__).resolve().parents[3] / "src/limite_vllm").glob("*.py"))
        },
        "enforce_eager": args.eager,
        "evaluation_kind": "diagnostic_non_aime" if args.diagnostic else "full_context_aime",
        "max_num_batched_tokens": args.max_batched_tokens,
        "profile": args.profile,
        "warmup_batch_sizes": warmup_sizes,
        "warmup_tokens_per_request": 16,
    }
    write(output / "manifest.json", contract)
    spec = next(s for s in suites.MATH_EXTENDED if s.name == "aime26")
    bindings = tasksets.taskset_protocols(spec)
    initialization_start = time.perf_counter()
    llm = LLM(
        model=str(root / "models/target"),
        tokenizer=str(root / "models/target"),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=config["max_position_embeddings"],
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_batched_tokens,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=False,
        enforce_eager=args.eager,
        seed=args.seed,
        disable_log_stats=False,
        async_scheduling=None,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str((output / "profile").resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
        }
        if args.profile
        else None,
        worker_extension_cls="limite_vllm.belluxx_v2.BelluxxV2WorkerExtension",
        speculative_config={
            "method": "custom_class",
            "model": "limite_vllm.belluxx_v2.BelluxxSpeculator",
            "num_speculative_tokens": args.draft_tokens,
        }
        if args.draft_tokens
        else None,
    )
    engine = llm.llm_engine
    contract["resolved_async_scheduling"] = engine.vllm_config.scheduler_config.async_scheduling
    contract["resolved_v2_runner"] = engine.vllm_config.use_v2_model_runner
    if bool(contract["resolved_v2_runner"]) != (args.model_runner == "v2"):
        raise RuntimeError("Resolved runner does not match requested runner")
    write(output / "manifest.json", contract)
    identity_workers = llm.collective_rpc("belluxx_runtime_identity")
    expected_runner = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
    if not all(worker["runner"] == expected_runner for worker in identity_workers):
        raise RuntimeError("Worker is not running the native V2 model runner")
    if args.draft_tokens and not all(
        worker["speculator"] == "limite_vllm.belluxx_v2.BelluxxSpeculator"
        and worker["auxiliary_outputs"]
        for worker in identity_workers
    ):
        raise RuntimeError("Worker is not using the Belluxx V2 speculator and features")
    write(output / "worker-identity.json", identity_workers)
    initialization_s = time.perf_counter() - initialization_start
    warmup_start = time.perf_counter()
    # Cover the padded batch shapes used as requests finish. Both arms use the
    # same warmup, whose cost is retained separately from measured evaluation.
    for batch_size in warmup_sizes:
        llm.generate(
            [{"prompt_token_ids": inputs[0]["prompt_token_ids"]}] * batch_size,
            SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16, seed=1741, ignore_eos=True),
            use_tqdm=False,
        )
    write(
        output / "startup.json",
        {
            "engine_initialization_s": initialization_s,
            "explicit_warmup_s": time.perf_counter() - warmup_start,
            "warmup_batch_sizes": warmup_sizes,
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
        },
    )
    if args.draft_tokens:
        llm.collective_rpc("reset_drafter_stats")
    if args.profile:
        llm.start_profile()
    active = {}
    completed = 0
    start = last_log = time.perf_counter()
    # Persist each completed request before admitting another, so an interruption
    # leaves independently inspectable samples rather than losing the whole run.
    with (output / "samples.jsonl").open("x", buffering=1) as journal:
        while pending or engine.has_unfinished_requests():
            while pending and len(active) < args.concurrency:
                record, rollout = pending.pop(0)
                key = f"{record['problem_index']}-{rollout}"
                params = SamplingParams(
                    temperature=args.temperature,
                    top_p=0.95,
                    top_k=-1,
                    max_tokens=record["max_tokens"],
                    seed=args.seed + record["problem_index"] * args.rollouts + rollout,
                    stop_token_ids=stop,
                    skip_special_tokens=False,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                    ignore_eos=args.diagnostic and not args.diagnostic_stop_at_eos,
                    logprobs=args.diagnostic_logprobs or None,
                )
                engine.add_request(key, {"prompt_token_ids": record["prompt_token_ids"]}, params)
                active[key] = (record, rollout, time.perf_counter())
            for result in engine.step():
                if not result.finished:
                    continue
                record, rollout, request_start = active.pop(result.request_id)
                generated = result.outputs[0]
                # Bounded validation is excluded from AIME quality and tuning must
                # not use gold answers. Only full-context runs invoke the grader.
                verdicts = {} if args.diagnostic else score(generated.text, record["gold"], bindings)
                row = {
                    "problem_index": record["problem_index"],
                    "rollout_index": rollout,
                    "gold": record["gold"],
                    "completion": generated.text,
                    "token_ids": list(generated.token_ids),
                    "completion_tokens": len(generated.token_ids),
                    "prompt_tokens": len(record["prompt_token_ids"]),
                    "max_tokens": record["max_tokens"],
                    "finish_reason": generated.finish_reason,
                    "stop_reason": generated.stop_reason,
                    "latency_s": time.perf_counter() - request_start,
                    "completed_at_s": time.perf_counter() - start,
                    "engine_metrics": dataclasses.asdict(result.metrics) if result.metrics else None,
                    "protocols": {k: dataclasses.asdict(v) for k, v in verdicts.items()},
                }
                if args.diagnostic_logprobs:
                    row["logprobs"] = [
                        {token: dataclasses.asdict(value) for token, value in position.items()}
                        for position in generated.logprobs
                    ]
                journal.write(json.dumps(row, default=str) + "\n")
                completed += 1
                info(
                    f"{completed}/{contract['expected_samples']} finished: problem {record['problem_index']} rollout {rollout}, {len(generated.token_ids)} tokens"
                )
            if time.perf_counter() - last_log > 30:
                info(
                    f"{completed}/{contract['expected_samples']} complete, {len(active)} active, elapsed {time.perf_counter() - start:.0f}s"
                )
                last_log = time.perf_counter()
    elapsed = time.perf_counter() - start
    if args.profile:
        llm.stop_profile()
    assert completed == contract["expected_samples"]
    if args.draft_tokens:
        write(output / "draft-stats-final.json", llm.collective_rpc("finish_drafter"))
    if args.model_runner == "v2":
        write(output / "runtime-stats.json", llm.collective_rpc("finish_runtime"))
    write(
        output / "completion.json",
        {
            "complete": True,
            "samples": completed,
            "elapsed_s": elapsed,
            "samples_sha256": digest(output / "samples.jsonl"),
        },
    )
    info(f"Completed {completed} samples in {elapsed:.1f}s")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, help="write new runs here; input root remains read-only")
    p.add_argument("--run-id", default="baseline")
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-batched-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--profile", action="store_true", help="bounded diagnostics only; alters timing")
    p.add_argument(
        "--diagnostic", action="store_true", help="Non-AIME synthetic prompts; bounded tuning only"
    )
    p.add_argument(
        "--diagnostic-stop-at-eos",
        action="store_true",
        help="Honor native EOS in bounded timing diagnostics",
    )
    p.add_argument(
        "--diagnostic-repeat",
        type=int,
        default=1,
        help="Repeat synthetic prompt for chunked-prefill checks",
    )
    p.add_argument(
        "--diagnostic-logprobs",
        type=int,
        default=0,
        help="Retain top token logprobs for numeric parity diagnostics",
    )
    p.add_argument(
        "--diagnostic-context-length",
        type=int,
        help="Lower context limit for bounded scheduler boundary checks",
    )
    p.add_argument("--drafter-profile", choices=["single-stream", "multistream"])
    p.add_argument("--draft-tokens", type=int, default=0)
    p.add_argument("--model-runner", choices=["v2"], default="v2")
    p.add_argument("--drafter-source", type=Path)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, help="bounded validation only; omit for AIME")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument(
        "--check",
        action="store_true",
        help="validate arguments and local paths without importing ML libraries",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.drafter_profile:
        count = {"single-stream": 5, "multistream": 3}[args.drafter_profile]
        if args.draft_tokens not in (0, count):
            raise ValueError("draft-tokens conflicts with drafter-profile")
        args.draft_tokens = count
    if (
        not 1 <= args.concurrency <= 32
        or args.rollouts < 1
        or args.limit is not None
        and not 1 <= args.limit <= 30
    ):
        raise ValueError("Invalid concurrency, rollouts, or problem limit")
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be one nonempty path component")
    if not 0 <= args.draft_tokens <= 7 or args.temperature < 0:
        raise ValueError("Invalid draft length or temperature")
    if args.diagnostic_repeat < 1 or (args.diagnostic_repeat != 1 and not args.diagnostic):
        raise ValueError("diagnostic-repeat requires a positive diagnostic workload")
    if args.diagnostic and args.max_new_tokens is None:
        raise ValueError("Diagnostic workloads require an explicit bounded output budget")
    if not 0 <= args.diagnostic_logprobs <= 20 or (args.diagnostic_logprobs and not args.diagnostic):
        raise ValueError("diagnostic-logprobs requires a diagnostic workload and 0..20 entries")
    if args.diagnostic_context_length is not None and (
        not args.diagnostic or args.diagnostic_context_length < 16
    ):
        raise ValueError(
            "diagnostic-context-length requires a diagnostic workload and at least 16 tokens"
        )
    if (
        args.diagnostic_context_length is not None
        and args.max_batched_tokens > args.diagnostic_context_length
    ):
        raise ValueError("Lower-context diagnostics require max-batched-tokens <= context length")
    if args.max_new_tokens is not None and not args.diagnostic:
        raise ValueError("Bounded runs must use non-AIME diagnostic prompts")
    if args.diagnostic_stop_at_eos and not args.diagnostic:
        raise ValueError("diagnostic-stop-at-eos requires a diagnostic workload")
    if args.max_batched_tokens < args.concurrency * (args.draft_tokens + 1):
        raise ValueError("max-batched-tokens must fit one complete decode batch")
    if not 1 <= args.shards <= 120 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard selection")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.profile and args.max_new_tokens is None:
        raise ValueError("Profiling requires a bounded diagnostic output budget")
    if args.draft_tokens and (
        args.drafter_source is None or not (args.drafter_source / "tools/spec/drafter.py").is_file()
    ):
        raise ValueError("A pinned drafter-source checkout is required")
    if args.phase == "run":
        for name in ["identity.json", "inputs.json", "models/target/model.safetensors"]:
            if not (args.root / name).is_file():
                raise FileNotFoundError(args.root / name)
        output = (args.output_root or args.root / "runs") / args.run_id
        if output.exists():
            raise FileExistsError(output)
        if args.draft_tokens and not (args.root / "models/drafter/model.safetensors").is_file():
            raise FileNotFoundError(args.root / "models/drafter/model.safetensors")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be present; its value is never printed")
    if args.check:
        print(
            "Arguments, required environment names, and referenced paths validated; no models or datasets loaded."
        )
        return
    prepare(args.root) if args.phase == "prepare" else run(args)


if __name__ == "__main__":
    main()
