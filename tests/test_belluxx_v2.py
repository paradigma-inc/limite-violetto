"""GPU correctness checks for accepted history, independent of AIME answers."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from limite_vllm.belluxx_v2 import FeatureRing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires compute GPU")
@torch.inference_mode()
def test_cached_projections_preserve_released_forward():
    from safetensors.torch import load_file

    from limite_vllm.belluxx import DraftForward, reference_module
    from limite_vllm.belluxx_cache import CachedDraftForward, ProjectContext

    reference = reference_module()
    state = load_file(
        str(Path(os.environ["LIMITE_BELLUXX_WEIGHTS"]) / "model.safetensors")
    )
    model = reference.Drafter(state["vocab"]).float().cuda().eval()
    model.load_state_dict(state, strict=True)
    torch.manual_seed(1741)
    context = reference.rms(torch.randn(4, 1024, 1280, device="cuda"))
    origins = torch.tensor([-1007, -637, 4096, 65536], device="cuda")
    positions = origins[:, None] + torch.arange(1024, device="cuda")
    context[positions < 0] = 0
    bonus = reference.rms(torch.randn(4, 1, 1280, device="cuda"))
    project = ProjectContext(model, reference)
    # Project in decode-sized chunks to test that changing the GEMM row shape
    # retains proposal parity with the original full-window forward.
    flat = context.flatten(0, 1)
    flat_pos = positions.flatten()
    projected = torch.cat(
        [project(flat[i : i + 4], flat_pos[i : i + 4]) for i in range(0, len(flat), 4)]
    ).reshape(4, 1024, 1024)
    for rows in (2, 4, 6, 8):
        expected = DraftForward(model, reference, rows)(context, origins, bonus)
        forward = CachedDraftForward(model, reference, rows)
        if rows in (4, 6):
            forward = torch.compile(forward, dynamic=False)
        actual = forward(context[:, -1], projected, origins, bonus)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires compute GPU")
@pytest.mark.parametrize("width", [1280, 1024])
@torch.inference_mode()
def test_history_survives_rejection_chunking_reordering_and_slot_reuse(width):
    from limite_vllm.belluxx_cache import last_accepted

    ring = FeatureRing(4, "cuda", width=width)
    oracle = {}
    # Slot, first position, scheduled length, rejected suffix. Long chunks wrap
    # the ring; mixed batches reorder slots; restarting at zero reuses a slot.
    steps = [
        [(0, 0, 1300, 0), (2, 0, 17, 0)],
        [(2, 17, 7, 5), (0, 1300, 8, 2)],
        [(3, 0, 1200, 0), (0, 1306, 4, 3), (2, 19, 8, 0)],
        [(0, 0, 5, 0), (3, 1200, 1, 0)],
        [(0, 5, 1500, 0)],
    ]
    expected_checks = 0
    for rows in steps:
        lengths = np.array([row[2] for row in rows], dtype=np.int32)
        starts = np.concatenate(([0], np.cumsum(lengths)))
        positions = torch.cat(
            [
                torch.arange(first, first + length, device="cuda")
                for _, first, length, _ in rows
            ]
        )
        features = torch.randn(int(lengths.sum()), width, device="cuda")
        batch = SimpleNamespace(
            num_reqs=len(rows),
            num_scheduled_tokens=lengths,
            query_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int32),
            positions=positions,
            idx_mapping=torch.tensor([row[0] for row in rows], device="cuda"),
            cu_num_logits=torch.arange(len(rows) + 1, device="cuda", dtype=torch.int32)
            * 8,
        )
        rejected = torch.tensor(
            [row[3] for row in rows], device="cuda", dtype=torch.int32
        )
        sampled = torch.tensor(
            [8 - row[3] for row in rows], device="cuda", dtype=torch.int32
        )
        ring.ingest(features, batch, rejected, sampled)
        context, origins, tokens = ring.gather(
            batch, torch.arange(4, device="cuda") + 10, 4
        )
        if width == 1280:
            last = last_accepted(
                features, batch, rejected, torch.empty(4, 1280, device="cuda")
            )
            torch.testing.assert_close(last, context[:, -1], rtol=0, atol=0)
        for i, (slot, first, length, reject) in enumerate(rows):
            accepted = features[starts[i] : starts[i] + length - reject]
            previous = oracle[slot] if first else features[:0]
            oracle[slot] = torch.cat([previous, accepted])[-1024:]
            expected = torch.zeros(1024, width, device="cuda")
            expected[-len(oracle[slot]) :] = oracle[slot]
            torch.testing.assert_close(context[i], expected, rtol=0, atol=0)
            assert origins[i].item() == first + length - reject - 1024
            assert tokens[i].item() == slot + 10
        torch.testing.assert_close(
            context[len(rows) :], torch.zeros_like(context[len(rows) :])
        )
        expected_checks += len(rows)
    assert ring.stats[0].item() == expected_checks
    assert ring.stats[1].item() == 0
    assert ring.ring.dtype == torch.float32


@pytest.mark.skipif(
    os.environ.get("RUN_BELLUXX_V2_ENGINE_TEST") != "1",
    reason="explicit bounded V2 engine validation",
)
@pytest.mark.parametrize("draft_tokens", [3, 5])
def test_engine_chunked_prefill_slot_reuse_and_context_limit(draft_tokens):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from limite_vllm.belluxx_v2 import CLASS_PATH

    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"
    target = Path(os.environ["LIMITE_TARGET_WEIGHTS"])
    tokenizer = AutoTokenizer.from_pretrained(
        target, trust_remote_code=True, token=os.environ["HF_TOKEN"]
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "List integers starting from one. "
                    + "Continue carefully. " * repeats,
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        for repeats in [2, 70, 3, 65, 4, 60]
    ]
    assert min(map(len, prompts)) < 128 < max(map(len, prompts)) < 512
    budgets = [512 - len(prompt) for prompt in prompts]
    llm = LLM(
        model=str(target),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        worker_extension_cls="limite_vllm.belluxx_v2.BelluxxV2WorkerExtension",
        speculative_config={
            "method": "custom_class",
            "model": CLASS_PATH,
            "num_speculative_tokens": draft_tokens,
        },
    )
    assert llm.llm_engine.vllm_config.use_v2_model_runner
    assert llm.llm_engine.vllm_config.scheduler_config.async_scheduling
    # Engine initialization also executes synthetic requests. Isolate the six
    # measured requests, just as the evaluation driver does after warmup.
    llm.collective_rpc("reset_drafter_stats")
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        [
            SamplingParams(
                temperature=0.6, top_p=0.95, seed=i, ignore_eos=True, max_tokens=budget
            )
            for i, budget in enumerate(budgets)
        ],
        use_tqdm=False,
    )
    assert [len(output.outputs[0].token_ids) for output in outputs] == budgets
    assert all(output.outputs[0].finish_reason == "length" for output in outputs)
    stats = llm.collective_rpc("finish_drafter")[0]
    assert stats["accepted"] > 0 and stats["continuity_errors"] == 0
    assert stats["context_checks"] > stats["target_steps"]
    assert stats["restarts"] == len(prompts)
    print({"bounded_context": 512, "completed_token_budgets": budgets, **stats})
