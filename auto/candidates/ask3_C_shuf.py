"""ask3 (C): J++ + lam * shuffled-target E_t3 control, lam chosen on FIT for the real E_t3 map."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, choose_lam, fit_line, load_maps, t3_path  # noqa: E402


def build(ctx):
    lam = choose_lam(ctx, load_maps(t3_path(ctx), ctx.device))
    lens = ELens(ctx.JPP, load_maps(t3_path(ctx, shuf=True), ctx.device), "mix", lam)
    fit_line(f"shuf mix {lam}", ctx, lens)
    return lens
