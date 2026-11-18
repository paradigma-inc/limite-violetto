"""Belluxx parallel draft proposals for the pinned vLLM 0.28 target runtime.

Uses the released Python drafter without retraining. vLLM performs exact
rejection sampling for deterministic proposals and owns target KV rollback.
Only accepted target features enter the 1024-position draft context. This is
a CUDA chain implementation, not the author's Metal tree implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

REFERENCE_SHA256 = "72dd976b2f4ec4cfdc0cffff9fc9c9b4631948bcb7b3168d06cd201c65ff6c92"


def reference_module():
    path = Path(os.environ["LIMITE_BELLUXX_SOURCE"]) / "tools/spec/drafter.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("Belluxx reference differs from the pinned released source")
    spec = importlib.util.spec_from_file_location("belluxx_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DraftForward(torch.nn.Module):
    """Reference forward with left-padding masked for batched prompt lengths."""

    def __init__(self, draft, reference, rows):
        super().__init__()
        self.draft = draft
        self.reference = reference
        self.rows = rows

    def forward(self, ctx, c0, bonus):
        r, d, rows = self.reference, self.draft, self.rows
        bs = ctx.shape[0]
        off = torch.full((bs, 1), 1024, device=ctx.device, dtype=torch.long)
        x = torch.cat([bonus[:, :, None], d.mask_emb.expand(bs, 1, rows - 1, r.H)], 2)
        x = (x + d.inject(ctx[:, -1:])[:, :, None]).reshape(bs, rows, r.H)
        pos = c0[:, None] + 1024 + torch.arange(rows, device=ctx.device)
        cos_b, sin_b = (t[:, None] for t in r.rope_tables(pos))
        context_pos = c0[:, None] + torch.arange(1024, device=ctx.device)
        cos_c, sin_c = (t[:, None] for t in r.rope_tables(context_pos))
        mask = r.block_mask(off, 1024, rows) & (context_pos[:, None] >= 0)
        for layer in d.layers:
            x = layer(x, layer.kv(ctx, cos_c, sin_c), cos_b, sin_b, mask, rows)
        hidden = r.rms(x) * d.out_gain
        return d.vocab[d.head(hidden[:, 1:]).argmax(-1)]
