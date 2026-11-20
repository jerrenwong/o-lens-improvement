"""g1 (AG): unit(J++ h) + lam * unit( unit(E h + b_E) + unit(G_strict h + b_G) ). E = ask-distilled map (runs/e1),
G_strict = generate-without-naming map (runs/g1); both learned maps, not Jacobians. lam chosen on FIT."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lib import AGLens, choose_lam, g_path, load_maps, map_path  # noqa: E402


def build(ctx):
    E = load_maps(map_path(ctx), ctx.device)
    G = load_maps(g_path("strict"), ctx.device)
    lam = choose_lam(ctx, lambda x: AGLens(ctx.JPP, E, G, x), "AG mix")
    return AGLens(ctx.JPP, E, G, lam)
