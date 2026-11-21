"""ask5 shared code (not a candidate). E_s5: the ask3 ridge ask-map recipe refit on SPECIFIC-unstated-word targets
elicited with the ask3 chat-template construction (scripts/ask5/ask5_collect.py, ask5_fit.py).
Maps .hf27/ask5/E_<tag>_s5(_shuf).pt; E_topic = .hf27/ask3/E_<tag>_t3.pt. Lens (fm27 Base form):
  v = unit(JPP h) + l1 unit(E_topic h + b) + l2 unit(E_spec h + b);  'alone': unit(E_spec h + b).
Selection (fixed before any dev run): argmax FIT (pass@1 + pass@10), filtered, ties -> first in grid order
(closest to J++), via fm27_lib.fit_eval_lens / choose."""
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps, t3_path  # noqa: E402
from fm27_lib import choose, excl_mask, fit_eval_lens, fmt, unit  # noqa: E402

R = "/workspace/o-lens-improvement"
D5 = os.environ.get("ASK5_MAPDIR", f"{R}/.hf27/ask5")  # override only for the smoke test
TAGS = {4096: "9b", 5120: "27b"}
GRID_A = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
GRID_C = [0.0, 0.1, 0.2, 0.3, 0.5]


def s5_path(ctx, shuf=False):
    return f"{D5}/E_{TAGS[ctx.model.d_model]}_s5{'_shuf' if shuf else ''}.pt"


class Lens:
    def __init__(self, JPP, maps_t, maps_s, l1, l2, alone=False):
        self.JPP, self.mt, self.ms, self.l1, self.l2, self.alone = JPP, maps_t, maps_s, l1, l2, alone

    @staticmethod
    def E(maps, h, l):
        mx, W, my = maps[l]
        return unit((h - mx) @ W + my)

    def transport(self, h, layer, resid_all=None):
        if self.alone:
            return self.E(self.ms, h, layer)
        v = unit(h @ self.JPP[layer].T)
        if self.l1:
            v = v + self.l1 * self.E(self.mt, h, layer)
        if self.l2:
            v = v + self.l2 * self.E(self.ms, h, layer)
        return v


def choose_a(ctx, maps_s, tag="a"):
    """lam for J++ + lam E_spec on FIT; cached to .hf27/ask5/lam_a_<tag>.json for (d)."""
    mask = excl_mask(ctx)
    res = []
    for lam in GRID_A:
        s = fit_eval_lens(ctx, Lens(ctx.JPP, None, maps_s, 0.0, lam), mask)
        print(fmt(f"ask5 {tag} lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[ask5 {tag}] chosen lam {lam}", flush=True)
    return lam


def lam_a(ctx):
    p = f"{D5}/lam_a_{TAGS[ctx.model.d_model]}.json"
    if os.path.exists(p):
        return json.load(open(p))["lam"]
    lam = choose_a(ctx, load_maps(s5_path(ctx), ctx.device))
    json.dump(dict(lam=lam), open(p, "w"))
    return lam


def choose_c(ctx, maps_t, maps_s):
    mask = excl_mask(ctx)
    res = []
    for l1, l2 in itertools.product(GRID_C, GRID_C):
        s = fit_eval_lens(ctx, Lens(ctx.JPP, maps_t, maps_s, l1, l2), mask)
        print(fmt(f"ask5 c l1 {l1} l2 {l2}", s), flush=True)
        res.append(((l1, l2), s))
    best = choose(res)
    print(f"[ask5 c] chosen (l1, l2) {best}", flush=True)
    return best
