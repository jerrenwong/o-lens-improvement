"""xm1 control: shuffled-answer map at L's own lam: unit(J++ h) + lam_L unit(E_shuf h + b_shuf). lam_L = the value
xm1_L chooses (same FIT selection recomputed here with the MAIN map; printed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xm1_lib import XBase, choose_lam, excl_mask, fit_eval_lens, fmt  # noqa: E402


def build(ctx):
    lam = choose_lam(ctx, "xm1 L (for Lshuf)")
    lens = XBase(ctx, lam, shuf=True)
    print(fmt(f"xm1 Lshuf lam{lam}", fit_eval_lens(ctx, lens, excl_mask(ctx))), flush=True)
    return lens
