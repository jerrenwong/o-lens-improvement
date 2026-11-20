"""g1 (G_shuf control): unit(J++ h) + lam * unit(G_shuf h + b), G_shuf = same ridge as G_strict with the concept
targets permuted across sentences. lam = the one chosen on FIT for g1_G (G_strict), recomputed here identically."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lib import ELens, choose_lam, fit_line, g_path, load_maps  # noqa: E402


def build(ctx):
    main = load_maps(g_path("strict"), ctx.device)
    lam = choose_lam(ctx, lambda x: ELens(ctx.JPP, main, "mix", x), "G_strict mix")
    del main
    maps = load_maps(g_path("shuf"), ctx.device)
    lens = ELens(ctx.JPP, maps, "mix", lam)
    fit_line("G_shuf", ctx, lens)
    return lens
