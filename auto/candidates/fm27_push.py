"""fm27 shared code (not a candidate): strict top-1 push used by F3 / F5a.
Fix over the first version (fm27_lib.F3Lens): the pushed token must be STRICTLY above every other filtered token
in the harness's bf16 logits; with '>= all others' a tie at bf16 resolution left the previous top-1 (often the
concept) also at rank 1, because harness ranks favour the lens on ties."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import NEG, F3Lens, logits, unit  # noqa: E402


class StrictPush(F3Lens):
    def ok(self, v, t):
        L = logits(self.model, v).masked_fill(self.mask, NEG)
        lt = L.gather(1, t[:, None])
        return (L >= lt).sum(-1) == 1

    def target(self, uj, ue, v):
        raise NotImplementedError

    def transport(self, h, layer, resid_all=None):
        uj, ue = self.parts(h, layer)
        v = uj + self.lam * ue
        t = self.target(uj, ue, v)
        bad = (~self.ok(v, t)).nonzero().flatten()
        if len(bad) == 0:
            return v
        d = unit(self.Wh[t[bad]].float() * self.gamma)
        vb, tb = v[bad], t[bad]
        lo = torch.zeros(len(bad), device=v.device)
        hi = torch.full((len(bad),), 0.25, device=v.device)
        for _ in range(14):
            good = self.ok(vb + hi[:, None] * d, tb)
            if good.all():
                break
            hi = torch.where(good, hi, hi * 2)
        good = self.ok(vb + hi[:, None] * d, tb)
        self.fallback += int((~good).sum())
        for _ in range(16):
            mid = (lo + hi) / 2
            g = self.ok(vb + mid[:, None] * d, tb)
            hi, lo = torch.where(g, mid, hi), torch.where(g, lo, mid)
        c = torch.where(good, hi, torch.zeros_like(hi))
        v = v.clone()
        v[bad] = vb + c[:, None] * d
        return v


class F3s(StrictPush):
    """C with J++'s own filtered top-1 strictly at rank 1."""

    def target(self, uj, ue, v):
        return logits(self.model, uj).masked_fill(self.mask, NEG).argmax(-1)


class F5s(StrictPush):
    """C, but rank 1 must be a token in J++'s filtered top-10: the J++-top-10 token with the highest C logit."""

    def target(self, uj, ue, v):
        j10 = logits(self.model, uj).masked_fill(self.mask, NEG).topk(10, dim=-1).indices
        LC = logits(self.model, v).masked_fill(self.mask, NEG)
        return j10.gather(1, LC.gather(1, j10).argmax(-1, keepdim=True))[:, 0]
