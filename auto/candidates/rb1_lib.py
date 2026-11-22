"""rb1 shared code (not a candidate). Lens L (fm27_F4 / st1_lib pattern: v = unit(JPP h) + lam unit(E h + b), lam
chosen on FIT from fm27_lib.GRID by max pass@1 + pass@10, ties -> smaller lam) with the ask-map swapped for an rb1
subset map .hf27/rb1/E_<tag>_<var>.pt (scripts/rb1/rb1_fit.py ridge refits, rb1_p3v.py P3v refits).
ALONE: the map read alone, v = unit(E h + b). Classes and build functions are st1_lib's (imported), only the path differs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import GRID, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402
from st1_lib import St1  # noqa: E402

R = "/workspace/o-lens-improvement"
TAGS = {2048: "2b", 4096: "9b", 5120: "27b"}


class Rb1(St1):
    def __init__(self, ctx, var, lam=0.0, alone=False):
        self.JPP = ctx.JPP
        self.maps = load_maps(f"{R}/.hf27/rb1/E_{TAGS[ctx.model.d_model]}_{var}.pt", ctx.device)
        self.lam, self.centred, self.alone = lam, False, alone


def build_L(ctx, var):
    mask = excl_mask(ctx)
    res = []
    for lam in GRID:
        s = fit_eval_lens(ctx, Rb1(ctx, var, lam), mask)
        print(fmt(f"rb1 {var} lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[rb1 {var}] chosen lam {lam}", flush=True)
    return Rb1(ctx, var, lam)


def build_alone(ctx, var):
    lens = Rb1(ctx, var, alone=True)
    print(fmt(f"rb1 {var} ALONE", fit_eval_lens(ctx, lens, excl_mask(ctx))), flush=True)
    return lens
