"""ask2 (b): J++ + lam * shuffled-target E_spec control; lam chosen on FIT for the real E_spec map (as e2_C_shuf)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask2_lib import ELens, choose_lam, fit_line, load_maps, spec_path  # noqa: E402


def build(ctx):
    lam = choose_lam(ctx, load_maps(spec_path(ctx), ctx.device), "spec")
    maps = load_maps(spec_path(ctx, shuf=True), ctx.device)
    lens = ELens(ctx.JPP, maps, "mix", lam)
    fit_line(f"shuf mix {lam}", ctx, lens)
    return lens
