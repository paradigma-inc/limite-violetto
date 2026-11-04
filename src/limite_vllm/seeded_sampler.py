"""vLLM 0.28.0 patch of the V2 runner's seeded sampler.

Requests that carry a ``seed`` (for example an evaluation run with ``--seed``)
send every decode step through vLLM's Triton top-p + Gumbel path. This patch
removes three costs from that path without changing a single sampled token or
logprob:

* the defensive fp32 copy of the logits, skipped when nothing in the batch
  consumes the original logits. The V2 runner drops logits after
  ``Sampler.__call__``; that method's only original logit consumers are
  logprobs and NaN diagnostics, and both keep the stock path;
* Gumbel noise for 1024-token blocks that top-p masked entirely
  (:mod:`limite_vllm.gumbel_masked`);
* the separate temperature pass, folded into top-p's first pass with the same
  fp32 division (:mod:`limite_vllm.temperature_topp`).

The upstream modules are pinned by source digest. ``Sampler.__init__`` runs the
original and patched samplers side by side on synthetic batches and refuses to
start unless token ids and logprobs are identical; host-side counters log how
often the patched path ran. No weight-derived state is retained, and all device
comparisons happen at startup.
"""

import copy
import hashlib
import inspect
import logging
import textwrap
import time
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace

_LOG = logging.getLogger(__name__)
_SOURCE_SHA = "5c06d57bf7df726146e385e678f8350bd2918fa29c7cfc233f12b81ba3e49f34"


def _eligible(self, logits, idx, return_logprobs):
    import numpy as np
    import torch

    return (
        not return_logprobs
        and not self.return_sampling_mask
        and not self.compute_nans
        and self.num_speculative_tokens == 1
        and logits.dtype == torch.float32
        and logits.is_cuda
        and logits.is_contiguous()
        and logits.ndim == 2
        and self.sampling_states.any_explicit_seed(idx)
        and not np.any(self.logit_bias_state.use_logit_bias[idx])
        and not np.any(self.penalties_state.use_penalty[idx])
        and not np.any(self.bad_words_state.num_bad_words.np[idx])
        and not self.thinking_budget_state.enabled
        and np.all(self.sampling_states.min_p.np[idx] == 0)
        and np.all(
            self.sampling_states.top_k.np[idx] == self.sampling_states.vocab_size
        )
        and np.any(self.needs_logits_processing[idx])
    )


def apply():
    import vllm.v1.worker.gpu.sample.sampler as upstream

    cls = upstream.Sampler
    if getattr(cls, "_limite_copy_patch", False):
        return
    if version("vllm").split("+", 1)[0] != "0.28.0":
        raise RuntimeError("sampler copy patch requires vLLM 0.28.0")
    if hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest() != _SOURCE_SHA:
        raise RuntimeError("sampler copy patch: upstream source changed")
    from limite_vllm import gumbel_masked, temperature_topp

    temperature_topp.check_source()
    gumbel_masked.check_source()
    original_sample = cls.sample
    original_init = cls.__init__
    # Derive the variant mechanically, retaining all upstream arithmetic and
    # dispatch verbatim. The source hash above pins the complete consumer code.
    source = textwrap.dedent(inspect.getsource(cls.apply_sampling_params))
    needle = "logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)"
    if source.count(needle) != 1:
        raise RuntimeError("sampler copy patch: fp32 copy not found exactly once")
    namespace = dict(vars(upstream))
    namespace["gumbel_sample"] = gumbel_masked.gumbel_sample
    exec(
        source.replace(needle, "pass  # caller proved original logits are unconsumed"),
        namespace,
    )
    no_copy_params = namespace["apply_sampling_params"]
    source = textwrap.dedent(inspect.getsource(original_sample))
    if source.count("self.apply_sampling_params(") != 1:
        raise RuntimeError(
            "sampler copy patch: apply_sampling_params call not found exactly once"
        )
    namespace["_no_copy_params"] = no_copy_params
    exec(
        source.replace("self.apply_sampling_params(", "_no_copy_params(self,"),
        namespace,
    )
    no_copy_sample = namespace["sample"]
    exec(source, namespace)
    combined_fallback_sample = namespace["sample"]

    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=False,
    ):
        use = _eligible(self, logits, idx_mapping_np, return_logprobs)
        if not getattr(self, "_copy_check_running", False):
            self._copy_counts[0 if use else 1] += 1
            now = time.monotonic()
            if now - self._copy_last_log >= 30:
                _LOG.warning(
                    "Limite sampler copy counters patched=%d fallback=%d",
                    *self._copy_counts,
                )
                self._copy_last_log = now
        fused = use and temperature_topp.eligible(self, logits, idx_mapping_np)
        if not getattr(self, "_copy_check_running", False):
            self._temperature_counts[0 if fused else 1] += 1
            now = time.monotonic()
            if now - self._temperature_last_log >= 30:
                _LOG.warning(
                    "Limite temperature fusion counters patched=%d fallback=%d",
                    *self._temperature_counts,
                )
                self._temperature_last_log = now
        if fused:
            # _eligible establishes that bias, penalties, bad words, thinking
            # budget, min-p and top-k are inactive and nobody needs raw logits.
            # top-p's host gather and RNG inputs are unchanged.
            _, top_p = self.sampling_states.get_top_k_top_p(
                expanded_idx_mapping, idx_mapping_np
            )
            processed = temperature_topp.apply(
                logits,
                top_p,
                self.sampling_states.temperature.gpu,
                expanded_idx_mapping,
            )
            sampled = gumbel_masked.gumbel_sample(
                processed,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )
            return sampled, processed
        fn = no_copy_sample if use else combined_fallback_sample
        return fn(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._copy_counts = [0, 0]
        self._temperature_counts = [0, 0]
        self._temperature_last_log = 0.0
        self._copy_last_log = 0.0
        # Constructor runs on the worker before profiling/serving, never forward.
        gumbel_masked._CHECKING = True
        try:
            _startup_check(self, original_sample, sample)
            with upstream.torch.inference_mode():
                gumbel_masked.check_blocks(self.sampling_states.temperature.gpu.device)
                temperature_topp.check_exact(
                    self.sampling_states.temperature.gpu.device,
                    self.sampling_states.vocab_size,
                )
        finally:
            gumbel_masked._CHECKING = False

    cls.sample = sample
    cls.__init__ = init
    cls._limite_copy_patch = True


def _startup_check(live, original, patched):
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
    from vllm.sampling_params import SamplingParams

    # Isolated synthetic request state; do not mutate live requests or global RNG.
    s = copy.copy(live)
    s._copy_check_running = True
    s.compute_nans = False
    s.return_sampling_mask = False
    s.num_speculative_tokens = 1
    s.use_flashinfer = False
    s.thinking_budget_state = SimpleNamespace(enabled=False, apply=lambda *a: None)
    s.logit_bias_state = SimpleNamespace(
        use_logit_bias=np.zeros(512, bool), apply_logit_bias=lambda *a: None
    )
    s.penalties_state = SimpleNamespace(
        use_penalty=np.zeros(512, bool), apply_penalties=lambda *a: None
    )
    s.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(np=np.zeros(512, np.int32)),
        apply_bad_words=lambda *a: None,
    )
    device = live.sampling_states.temperature.gpu.device
    vocab = live.sampling_states.vocab_size
    s.sampling_states = SamplingStates(512, vocab)
    s.logprob_token_ids_state = LogprobTokenIdsState(512, device)
    s.logprob_token_ids_state.num_token_ids.np.fill(0)
    s.logprob_token_ids_state.apply_staged_writes()
    s.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(
            gpu=torch.zeros(512, device=device, dtype=torch.int32)
        )
    )
    cases = 0
    selected = [0, 0]
    with torch.inference_mode():
        for rows in (1, 7, 255, 256, 257, 320, 512):
            idx_np = np.arange(rows, dtype=np.int32)[::-1].copy()
            idx = torch.tensor(idx_np, device=device)
            seq = torch.arange(rows, device=device, dtype=torch.int32)
            batch = SimpleNamespace(
                expanded_idx_mapping=idx,
                idx_mapping=idx,
                idx_mapping_np=idx_np,
                cu_num_logits_np=np.arange(rows + 1, dtype=np.int32),
                expanded_local_pos=torch.zeros_like(seq),
                positions=seq * 17 + 23,
                logits_indices=seq.to(torch.int64),
                input_ids=seq.to(torch.int64) % vocab,
                seq_lens=seq + 100,
                num_reqs=rows,
                cu_num_logits=torch.arange(rows + 1, device=device, dtype=torch.int32),
            )
            for peaked in (False, True):
                # Flat with small non-tied variation; peaked nucleus of one.
                col = torch.arange(vocab, device=device, dtype=torch.float32)
                logits = (0.1 + (col % 101) * 0.0001).expand(rows, -1).clone()
                if peaked:
                    logits[seq.long(), (seq.long() * 997 + 13) % vocab] = 22.9
                for temp in (0.6, 1.0):
                    for top_p in (0.95, 1.0):
                        for mode in (None, "raw_logprobs", "processed_logprobs"):
                            for i in range(rows):
                                # A mixed greedy row, while preserving stochastic
                                # single-row cases at both temperatures.
                                t = 0.0 if rows > 1 and i == rows - 1 else temp
                                s.sampling_states.add_request(
                                    i,
                                    SamplingParams(
                                        temperature=t,
                                        top_p=top_p,
                                        seed=12345 + i * 7919,
                                        logprobs=3 if mode and i % 2 == 0 else None,
                                    ),
                                )
                            s.sampling_states.apply_staged_writes()
                            st = s.sampling_states
                            s.needs_logits_processing = (
                                (st.temperature.np != 0) & (st.temperature.np != 1)
                            ) | (st.top_p.np != 1)
                            s.logprobs_mode = mode or "raw_logprobs"
                            a, b = logits.clone(), logits.clone()
                            selected[
                                int(not _eligible(s, b, idx_np, mode is not None))
                            ] += 1
                            s.sample = MethodType(original, s)
                            ref = s(a, batch)
                            s.sample = MethodType(patched, s)
                            got = s(b, batch)
                            if not torch.equal(
                                ref.sampled_token_ids, got.sampled_token_ids
                            ):
                                raise RuntimeError(
                                    f"sampler token mismatch: {rows, peaked, temp, top_p, mode}"
                                )
                            if mode:
                                for name in (
                                    "logprob_token_ids",
                                    "logprobs",
                                    "selected_token_ranks",
                                ):
                                    if not torch.equal(
                                        getattr(ref.logprobs_tensors, name),
                                        getattr(got.logprobs_tensors, name),
                                    ):
                                        raise RuntimeError(
                                            f"sampler logprob mismatch: {name}"
                                        )
                                if not torch.equal(a, b):
                                    raise RuntimeError(
                                        "fallback changed original logits"
                                    )
                            cases += 1
    if not all(selected):
        raise RuntimeError("startup check failed to exercise both paths")
    _LOG.warning(
        "Limite sampler startup equivalence PASS cases=%d patched=%d fallback=%d exact_ids_and_logprobs=True",
        cases,
        *selected,
    )
