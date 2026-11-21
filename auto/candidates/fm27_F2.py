"""fm27 F2 centred map: unit(JPP h) + lam unit(E h + b - b_prior), b_prior = mean target; lam chosen on FIT by
max pass@1 + pass@10 (ties -> smaller lam)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import GRID, Base, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402


def build(ctx):
    mask = excl_mask(ctx)
    res = []
    for lam in GRID:
        s = fit_eval_lens(ctx, Base(ctx, lam, centred=True), mask)
        print(fmt(f"F2 lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[fm27 F2] chosen lam {lam}", flush=True)
    return Base(ctx, lam, centred=True)
