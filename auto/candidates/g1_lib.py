"""Shared code for g1_* candidates (not a candidate itself).

G_l is a LEARNED MAP trained on model-generated sentences, not a Jacobian: a per-layer ridge map (with bias) from
the residual at the last token of a sentence that the model wrote about a concept word WITHOUT naming it, to the
concept's target vector t_w (mean unit unembedding row of its single-token surface forms, in the final basis t/g).
Label-free: concepts = 6000 frequent vocabulary words; STRICT variant removes every benchmark word
(scripts/g1_concepts.py, g1_gen.py, g1_collect.py, g1_fit.py). Candidates only LOAD the maps from runs/g1/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps, map_path  # noqa: E402,F401
from b2d_common import unit  # noqa: E402

RUNS = "/workspace/o-lens-improvement/runs/g1"


def g_path(variant):
    return f"{RUNS}/G_{variant}.pt"


def choose_lam(ctx, make, tag):
    base = fit_line("jpp", ctx, make(0.0))
    scores = {lam: fit_line(f"{tag} {lam}", ctx, make(lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[g1 fit] {tag}: chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return best


class AGLens:
    """unit(JPP h) + lam * unit( unit(E h + b_E) + unit(G h + b_G) )."""

    def __init__(self, JPP, E, G, lam):
        self.JPP, self.E, self.G, self.lam = JPP, E, G, lam

    @staticmethod
    def _m(maps, h, l):
        mx, W, my = maps[l]
        return (h - mx) @ W + my

    def transport(self, h, layer, resid_all=None):
        j = unit(h @ self.JPP[layer].T)
        a = unit(self._m(self.E, h, layer)) + unit(self._m(self.G, h, layer))
        return j + self.lam * unit(a)


def build_mix(ctx, variant):
    path = g_path(variant)
    print(f"[g1] maps {path}", flush=True)
    maps = load_maps(path, ctx.device)
    lam = choose_lam(ctx, lambda x: ELens(ctx.JPP, maps, "mix", x), f"G_{variant} mix")
    fit_line(f"G_{variant} alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
