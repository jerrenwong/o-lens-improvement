"""rb1 controls (not a candidate): lens L pattern with an rb1 control map at a lam taken from a REFERENCE map: the lam
the same FIT rule (fm27_lib.choose over fm27_lib.GRID, max pass@1 + pass@10) picks for the reference rb1 map
(e.g. ref "rall" = the stored map -> lens L's own lam). Classes from rb1_lib (imported, unchanged)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import GRID, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402
from rb1_lib import Rb1  # noqa: E402


def build_at(ctx, var, ref):
    mask = excl_mask(ctx)
    res = [(lam, fit_eval_lens(ctx, Rb1(ctx, ref, lam), mask)) for lam in GRID]
    lam = choose(res)
    print(f"[rb1 {var} at {ref}] reference lam {lam}", flush=True)
    lens = Rb1(ctx, var, lam)
    print(fmt(f"rb1 {var} lam{lam}", fit_eval_lens(ctx, lens, mask)), flush=True)
    print(f"[rb1 {var}] chosen lam {lam}", flush=True)
    return lens
