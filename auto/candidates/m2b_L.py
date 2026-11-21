"""m2b L: lens L (fm27_F4) for Qwen3.5-2B: lam re-chosen on the 2B FIT split from {0, .05, .1, .2, .3, .5, 1.0} by
max pass@1 + pass@10 (filtered; ties -> smaller lam); map .hf27/ask3/E_2b_t3.pt. PREREG_v4.md Addendum B."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2b_lib import GRID, Base2B, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402


def build(ctx):
    mask = excl_mask(ctx)
    res = []
    for lam in GRID:
        s = fit_eval_lens(ctx, Base2B(ctx, lam), mask)
        print(fmt(f"m2b L lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[m2b L] chosen lam {lam}", flush=True)
    return Base2B(ctx, lam)
