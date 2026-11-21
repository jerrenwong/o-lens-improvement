"""ask2 (d): E_spec alone (E h + b; learned map, not a Jacobian)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask2_lib import ELens, fit_line, load_maps, spec_path  # noqa: E402


def build(ctx):
    maps = load_maps(spec_path(ctx), ctx.device)
    lens = ELens(ctx.JPP, maps, "alone")
    fit_line("E_spec alone", ctx, lens)
    return lens
