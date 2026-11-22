"""xm1 control: constant-only map at L's own lam: unit(J++ h) + lam_L unit(b), b = the main map's mean output (same
vector for every prompt). lam_L = the value xm1_L chooses (same FIT selection recomputed here; printed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xm1_lib import choose_lam, excl_mask, fit_eval_lens, fmt, prior_lens  # noqa: E402


def build(ctx):
    lam = choose_lam(ctx, "xm1 L (for Lprior)")
    lens = prior_lens(ctx, lam)
    print(fmt(f"xm1 Lprior lam{lam}", fit_eval_lens(ctx, lens, excl_mask(ctx))), flush=True)
    return lens
