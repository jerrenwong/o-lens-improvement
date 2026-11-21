"""ask2 shared code (not a candidate). E_spec: the e2 ask-map recipe (LEARNED ridge map, not a Jacobian) with
SPECIFIC-unstated-word few-shot targets (scripts/ask2/ask2_collect.py, ask2_fit.py) and junk targets dropped.
Maps: .hf27/ask2/E_<tag>_spec.pt, E_<tag>_spec_shuf.pt; topic map (recipe A): runs/e1/E_<tag>.pt."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps  # noqa: E402,F401
from b2d_common import unit  # noqa: E402

R = "/workspace/o-lens-improvement"
TAGS = {4096: "9b", 5120: "27b"}


def tag(ctx):
    return TAGS[ctx.model.d_model]


def spec_path(ctx, shuf=False):
    return f"{R}/.hf27/ask2/E_{tag(ctx)}_spec{'_shuf' if shuf else ''}.pt"


def topic_path(ctx):
    return f"{R}/runs/e1/E_{tag(ctx)}.pt"


def choose_lam(ctx, maps, label):
    scores = {lam: fit_line(f"{label} mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[ask2 fit] {label} chosen lam {best} fit {scores[best]:.4f}", flush=True)
    return best


class Both:
    """unit(JPP h) + l1 unit(E_topic h + b) + l2 unit(E_spec h + b)."""

    def __init__(self, JPP, mt, ms, l1, l2):
        self.JPP, self.mt, self.ms, self.l1, self.l2 = JPP, mt, ms, l1, l2

    @staticmethod
    def E(m, h, l):
        mx, W, my = m[l]
        return (h - mx) @ W + my

    def transport(self, h, layer, resid_all=None):
        v = unit(h @ self.JPP[layer].T)
        if self.l1:
            v = v + self.l1 * unit(self.E(self.mt, h, layer))
        if self.l2:
            v = v + self.l2 * unit(self.E(self.ms, h, layer))
        return v
