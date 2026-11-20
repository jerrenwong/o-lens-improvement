"""g1 (Galone): G_strict h + b alone (learned map trained on model-generated sentences, not a Jacobian; no J++)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lib import ELens, fit_line, g_path, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps(g_path("strict"), ctx.device)
    lens = ELens(ctx.JPP, maps, "alone")
    fit_line("G_strict alone", ctx, lens)
    return lens
