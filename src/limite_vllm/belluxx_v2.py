"""Belluxx on vLLM 0.28 V2, with device-resident accepted feature history.

The target's standard rejection sampler owns verification and KV rollback.
Persistent V2 request slots index a 1024-position FP32 feature ring. Only the
accepted prefix is ingested, including chunked prefill and context exhaustion.
Prefix caching is deliberately unsupported: recomputation must begin at zero.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from limite_vllm.belluxx import reference_module
from limite_vllm.belluxx_config import CLASS_PATH, is_belluxx, validate_config
from limite_vllm.belluxx_cache import CachedDraftForward, ProjectContext, last_accepted

WINDOW, HIDDEN = 1024, 1280
COUNTERS = (
    "context_checks",
    "continuity_errors",
    "target_steps",
    "emitted",
    "proposed",
    "accepted",
    "restarts",
)


@triton.jit
def _ingest(
    fused,
    ring,
    ends,
    stats,
    starts,
    positions,
    mapping,
    rejected,
    sampled,
    logits,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(mapping + req)
    a, b = tl.load(starts + req), tl.load(starts + req + 1)
    first = tl.load(positions + a)
    valid = b - a - tl.load(rejected + req)
    end = first + valid
    kept = tl.minimum(valid, W)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    row, channel = offset // H, offset % H
    pos = end - kept + row
    value = tl.load(
        fused + (a + valid - kept + row) * H + channel, mask=row < kept, other=0
    )
    tl.store(ring + (slot * W + pos % W) * H + channel, value, mask=row < kept)
    if tile == 0:
        previous = tl.load(ends + slot)
        bad = ((first != 0) & (first != previous)) | (valid <= 0)
        tl.atomic_add(stats + 0, 1)
        tl.atomic_add(stats + 1, bad.to(tl.int64))
        count = tl.load(sampled + req)
        proposed = tl.load(logits + req + 1) - tl.load(logits + req) - 1
        tl.atomic_add(stats + 2, (count > 0).to(tl.int64))
        tl.atomic_add(stats + 3, count.to(tl.int64))
        tl.atomic_add(stats + 4, tl.where(count > 0, proposed, 0).to(tl.int64))
        tl.atomic_add(stats + 5, tl.maximum(count - 1, 0).to(tl.int64))
        tl.atomic_add(stats + 6, (first == 0).to(tl.int64))
        tl.store(ends + slot, end)


@triton.jit
def _gather(
    ring,
    ends,
    context,
    origins,
    tokens,
    last_sampled,
    mapping,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req, tile = tl.program_id(0), tl.program_id(1)
    active = req < N
    slot = tl.load(mapping + req, mask=active, other=0)
    end = tl.load(ends + slot, mask=active, other=0)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    pos, channel = end - W + offset // H, offset % H
    value = tl.load(
        ring + (slot * W + pos % W) * H + channel,
        mask=active & (pos >= 0) & (offset < W * H),
        other=0,
    )
    tl.store(context + req * W * H + offset, value, mask=offset < W * H)
    if tile == 0:
        tl.store(origins + req, end - W)
        token = tl.load(last_sampled + slot, mask=active, other=0)
        tl.store(tokens + req, token)


class FeatureRing:
    def __init__(self, slots, device, width=HIDDEN):
        self.width = width
        self.ring = torch.zeros(
            slots, WINDOW, width, device=device, dtype=torch.float32
        )
        self.ends = torch.zeros(slots, dtype=torch.int64, device=device)
        self.stats = torch.zeros(len(COUNTERS), dtype=torch.int64, device=device)
        self.context = torch.zeros_like(self.ring)
        self.origins = torch.zeros_like(self.ends)
        self.tokens = torch.zeros_like(self.ends)

    def ingest(self, fused, batch, num_rejected, num_sampled):
        max_rows = min(int(batch.num_scheduled_tokens.max()), WINDOW)
        _ingest[(batch.num_reqs, triton.cdiv(max_rows * self.width, 1024))](
            fused,
            self.ring,
            self.ends,
            self.stats,
            batch.query_start_loc,
            batch.positions,
            batch.idx_mapping,
            num_rejected,
            num_sampled,
            batch.cu_num_logits,
            self.width,
            WINDOW,
            1024,
        )
        # Device-side assertion; no host synchronization in the generation loop.
        torch._assert_async(
            self.stats[1] == 0, "Belluxx accepted-feature discontinuity"
        )

    def gather(self, batch, last_sampled, padded):
        _gather[(padded, triton.cdiv(WINDOW * self.width, 1024))](
            self.ring,
            self.ends,
            self.context,
            self.origins,
            self.tokens,
            last_sampled,
            batch.idx_mapping,
            batch.num_reqs,
            self.width,
            WINDOW,
            1024,
        )
        return self.context[:padded], self.origins[:padded], self.tokens[:padded]


class BelluxxSpeculator(DraftModelSpeculator):
    """Implements V2's draft-model lifecycle without allocating draft KV pages."""

    def __init__(self, vllm_config, device):
        self.vllm_config, self.device = vllm_config, device
        self.speculative_config = vllm_config.speculative_config
        self.k = self.speculative_config.num_speculative_tokens
        validate_config(vllm_config)
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.padded_slots = triton.next_power_of_2(self.max_num_reqs)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.supports_mm_inputs = False
        self.draft_logits = None
        self.draft_attn_layer_names = set()
        self.eplb_state = None
        self.calls = 0
        self.stats_path = os.environ.get("LIMITE_BELLUXX_STATS")

    def load_draft_model(self, target_model, target_attn_layer_names):
        raise RuntimeError("Belluxx owns its complete load_model lifecycle")

    def load_model(self, target_model):
        self.reference = reference_module()
        state = load_file(
            str(Path(os.environ["LIMITE_BELLUXX_WEIGHTS"]) / "model.safetensors")
        )
        # vLLM sets BF16 as the default dtype while loading target weights.
        # The released drafter and fusion must explicitly retain FP32.
        with torch.device("cpu"):
            self.model = self.reference.Drafter(state["vocab"]).float()
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.embed = target_model.embed_input_ids
        self.features = FeatureRing(self.padded_slots, self.device, width=1024)
        self.last = torch.zeros(
            self.padded_slots, HIDDEN, device=self.device, dtype=torch.float32
        )
        self.project = torch.compile(
            ProjectContext(self.model, self.reference), dynamic=True
        )
        self.forward = torch.compile(
            CachedDraftForward(self.model, self.reference, self.k + 1), dynamic=False
        )

    def set_attn(self, *args, **kwargs):
        # The fixed-window reference forward does not use vLLM attention/KV pages.
        pass

    def init_cudagraph_manager(self, cudagraph_mode):
        # Compilation uses default mode; target CUDA graphs remain independent.
        pass

    def capture(self):
        pass

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        n, count = input_batch.num_tokens, input_batch.num_reqs
        padded = triton.next_power_of_2(count)
        if dummy_run:
            # Dummy target positions are all zero, so never ingest them as history.
            context = self.features.context[:padded]
            origins = self.features.origins[:padded]
            tokens = self.features.tokens[:padded]
            last = self.last[:padded]
        else:
            assert aux_hidden_states is not None and len(aux_hidden_states) == 2
            features = torch.stack(
                [
                    self.reference.rms(h[:n].float()).bfloat16()
                    for h in aux_hidden_states
                ],
                -2,
            )
            fused = self.model.context(features[None])[0]
            projected = self.project(fused, input_batch.positions[:n])
            self.features.ingest(projected, input_batch, num_rejected, num_sampled)
            last = last_accepted(fused, input_batch, num_rejected, self.last[:padded])
            context, origins, tokens = self.features.gather(
                input_batch, last_sampled, padded
            )
            self.calls += 1
        bonus = self.reference.rms(self.embed(tokens).float())[:, None]
        proposals = self.forward(last, context, origins, bonus)[:count]
        # V2's scheduler clips scheduled drafts to the remaining context capacity.
        # Always ingest accepted features, even for requests that cannot draft.
        return proposals

    def reset_stats(self):
        self.features.stats.zero_()
        self.calls = 0

    def save_stats(self):
        stats = dict(zip(COUNTERS, self.features.stats.cpu().tolist()))
        stats.update(
            calls=self.calls,
            implementation=CLASS_PATH,
            feature_storage="GPU FP32 projected K/V ring",
            window=WINDOW,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        if stats["continuity_errors"]:
            raise RuntimeError(f"Invalid feature continuity: {stats}")
        if self.stats_path:
            path = Path(self.stats_path)
            path.write_text(json.dumps(stats, indent=2) + "\n")
        return stats


class BelluxxV2WorkerExtension:
    def belluxx_runtime_identity(self):
        runner = self.model_runner
        speculator = getattr(runner, "speculator", None)
        return {
            "runner": type(runner).__module__ + "." + type(runner).__name__,
            "runner_source": inspect.getfile(type(runner)),
            "speculator": type(speculator).__module__ + "." + type(speculator).__name__
            if speculator is not None
            else None,
            "auxiliary_outputs": runner.use_aux_hidden_state_outputs,
            "rejection_sampler": type(
                getattr(runner, "rejection_sampler", None)
            ).__module__,
            "allocated_bytes": torch.cuda.memory_allocated(),
        }

    def finish_runtime(self):
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        identity = self.belluxx_runtime_identity()
        return dict(
            **identity,
            runner_class=type(self.model_runner).__module__
            + "."
            + type(self.model_runner).__name__,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_uuid=str(properties.uuid),
            gpu_name=properties.name,
        )

    def reset_drafter_stats(self):
        self.model_runner.speculator.reset_stats()
        return True

    def finish_drafter(self):
        result = self.model_runner.speculator.save_stats()
        result["runner_class"] = (
            type(self.model_runner).__module__ + "." + type(self.model_runner).__name__
        )
        return result


def install():
    """Fail-closed source-pinned factory/config integration, installed per process."""
    import vllm.config.speculative as spec_module
    import vllm.config.vllm as config_module
    import vllm.v1.worker.gpu.model_runner as runner_module
    import vllm.v1.worker.gpu.spec_decode as factory_module
    from importlib.metadata import version

    config_cls = config_module.VllmConfig
    if getattr(config_cls, "_belluxx_v2_installed", False):
        return
    if version("vllm").split("+", 1)[0] != "0.28.0":
        raise RuntimeError("Belluxx V2 requires pinned vLLM 0.28.0")
    sources = {
        config_module: "fb07f95658ae0ee6e6c16c0f36e90f9179e8761221a45d4bf1aebe8bcf63cdcb",
        spec_module: "bb2955435856aa85633e7d799ce0ba241641952d65ac21ae10f097000ebd98f2",
        runner_module: "fb08762e3a025fdc55ab8d2605a96a57d1fad7efae0fbb605cc4fe2973028aca",
        factory_module: "22374cea4acdc10e92bc3a043503e51602f545b6f09ac7983c4a7515806a1675",
    }
    for module, expected in sources.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Belluxx V2 upstream source changed: {module.__name__}")

    original_unsupported = config_cls._get_v2_model_runner_unsupported_features

    def unsupported(self):
        result = original_unsupported(self)
        if is_belluxx(self.speculative_config):
            validate_config(self)
            result = [x for x in result if x != "speculative method 'custom_class'"]
        return result

    config_cls._get_v2_model_runner_unsupported_features = unsupported
    # Preserve every other async compatibility check verbatim. Only this exact
    # class may join the already-supported device-resident proposer methods.
    source = textwrap.dedent(inspect.getsource(config_cls.__post_init__))
    needle = "self.speculative_config.method not in get_args(EagleModelTypes)"
    if source.count(needle) != 2:
        raise RuntimeError("Belluxx V2 async compatibility checks changed")
    namespace = dict(vars(config_module), is_belluxx=is_belluxx)
    exec(
        source.replace(
            needle, f"({needle} and not is_belluxx(self.speculative_config))"
        ),
        namespace,
    )
    config_cls.__post_init__ = namespace["__post_init__"]
    original_eagle = spec_module.SpeculativeConfig.use_eagle
    spec_module.SpeculativeConfig.use_eagle = lambda self: (
        is_belluxx(self) or original_eagle(self)
    )
    original_factory = factory_module.init_speculator

    def factory(config, device):
        if is_belluxx(config.speculative_config):
            return BelluxxSpeculator(config, device)
        return original_factory(config, device)

    factory_module.init_speculator = runner_module.init_speculator = factory
    original_init = runner_module.GPUModelRunner.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if isinstance(self.speculator, BelluxxSpeculator):
            self.use_aux_hidden_state_outputs = True

    runner_module.GPUModelRunner.__init__ = init
    config_cls._belluxx_v2_installed = True
