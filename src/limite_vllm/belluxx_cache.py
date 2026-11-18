"""FP32 Belluxx context projections reusable across accepted decode steps."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _last_accepted(
    fused, starts, rejected, last, N, H: tl.constexpr, BLOCK: tl.constexpr
):
    req = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    end = tl.load(starts + req + 1, mask=req < N, other=1)
    reject = tl.load(rejected + req, mask=req < N, other=0)
    value = tl.load(
        fused + (end - reject - 1) * H + channel,
        mask=(req < N) & (channel < H),
        other=0,
    )
    tl.store(last + req * H + channel, value, mask=channel < H)


def last_accepted(fused, batch, rejected, output):
    """Gather each active request's last accepted fusion, zeroing padded rows."""
    _last_accepted[(len(output),)](
        fused, batch.query_start_loc, rejected, output, batch.num_reqs, 1280, 2048
    )
    return output


class ProjectContext(torch.nn.Module):
    def __init__(self, draft, reference):
        super().__init__()
        self.draft, self.reference = draft, reference

    def forward(self, fused, positions):
        cos, sin = (part[None, None] for part in self.reference.rope_tables(positions))
        # Token-major storage: [token, layer, key/value, head, channel]. Each
        # accepted token's independent projections are computed exactly once.
        layers = [
            torch.stack(layer.kv(fused[None], cos, sin), 0)
            for layer in self.draft.layers
        ]
        return torch.stack(layers, 0)[:, :, 0].permute(3, 0, 1, 2, 4).reshape(-1, 1024)


class CachedDraftForward(torch.nn.Module):
    def __init__(self, draft, reference, rows):
        super().__init__()
        self.draft, self.reference, self.rows = draft, reference, rows

    def forward(self, last, projected, c0, bonus):
        r, d, rows = self.reference, self.draft, self.rows
        bs = last.shape[0]
        off = torch.full((bs, 1), 1024, device=last.device, dtype=torch.long)
        x = torch.cat([bonus[:, :, None], d.mask_emb.expand(bs, 1, rows - 1, r.H)], 2)
        x = (x + d.inject(last[:, None])[:, :, None]).reshape(bs, rows, r.H)
        pos = c0[:, None] + 1024 + torch.arange(rows, device=last.device)
        cos_b, sin_b = (part[:, None] for part in r.rope_tables(pos))
        context_pos = c0[:, None] + torch.arange(1024, device=last.device)
        mask = r.block_mask(off, 1024, rows) & (context_pos[:, None] >= 0)
        kv = projected.view(bs, 1024, 2, 2, 2, 128)
        for i, layer in enumerate(d.layers):
            keys = kv[:, :, i, 0].transpose(1, 2)
            values = kv[:, :, i, 1].transpose(1, 2)
            x = layer(x, (keys, values), cos_b, sin_b, mask, rows)
        hidden = r.rms(x) * d.out_gain
        return d.vocab[d.head(hidden[:, 1:]).argmax(-1)]
