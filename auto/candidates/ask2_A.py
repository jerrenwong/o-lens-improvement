"""ask2 (a): J++ + lam * E_spec (specific-unstated-word ask-map; learned, not a Jacobian). lam on FIT as e2_A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask2_lib import ELens, choose_lam, fit_line, load_maps, spec_path  # noqa: E402


def build(ctx):
    print(f"[ask2] maps {spec_path(ctx)}", flush=True)
    maps = load_maps(spec_path(ctx), ctx.device)
    fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    lam = choose_lam(ctx, maps, "spec")
    fit_line("E_spec alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
