"""Shared code for al1_* candidates (not a candidate itself).

M_l is a LEARNED MAP, not a Jacobian: a per-layer linear "concept -> word" translation fitted by scripts/al1/fit.py on
2,000 vocabulary words (benchmark words excluded) x ~20 model-written sentences per word that never name the word.
Input: last-token residual of a sentence (run alone); target: the word's mean unit unembedding row (final basis).
The variant (sentence ridge / class-mean ridge / within-word-whitened ridge / reduced rank) and its options were
chosen on 200 separate validation words. Candidates only LOAD the maps (from .hf27/al1/).
Lens: unit(JPP_l h) + lam * unit(M_l h + bias).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps  # noqa: E402,F401

BIG = "/workspace/o-lens-improvement/.hf27/al1"


def m_path(variant="best"):
    return f"{BIG}/M_{variant}.pt"


def choose_lam(ctx, maps, tag):
    base = fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"{tag} {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[al1 fit] {tag}: chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return best
