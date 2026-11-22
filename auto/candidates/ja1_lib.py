"""ja1 shared code (not a candidate). J_ask lenses: plain dicts {layer: [d, d] fp32} written by scripts/ja1/ja1_post.py,
path in env JA1_J. Alone: h @ J^T. Added: unit(J++ h) + lam unit(J_ask h), lam chosen on FIT from fm27_lib.GRID by
max pass@1 + pass@10 (filtered; ties -> smaller lam), exactly the fm27_F4 rule (fm27_lib imported, not edited)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import GRID, choose, excl_mask, fit_eval_lens, fmt, unit  # noqa: E402


def load_J(ctx):
    M = torch.load(os.environ["JA1_J"], map_location="cpu", weights_only=True)
    M = {int(l): v.to(ctx.device).float() for l, v in M.items()}
    assert next(iter(M.values())).shape[0] == ctx.model.d_model, "J_ask file is for another model"
    return M


class Alone:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


class Added:
    def __init__(self, JPP, M, lam):
        self.JPP, self.M, self.lam = JPP, M, lam

    def transport(self, h, layer, resid_all=None):
        return unit(h @ self.JPP[layer].T) + self.lam * unit(h @ self.M[layer].T)


def build_alone(ctx):
    M = load_J(ctx)
    print(fmt("ja1 alone FIT", fit_eval_lens(ctx, Alone(M), excl_mask(ctx))), flush=True)
    return Alone(M)


def build_added(ctx):
    M, mask, res = load_J(ctx), excl_mask(ctx), []
    for lam in GRID:
        s = fit_eval_lens(ctx, Added(ctx.JPP, M, lam), mask)
        print(fmt(f"ja1 add lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[ja1 add] chosen lam {lam}", flush=True)
    return Added(ctx.JPP, M, lam)
