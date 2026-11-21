"""Shared code for al2_* candidates (not a candidate itself).

M_l is a LEARNED MAP, not a Jacobian: al1's noise-aware ridge (whiten by within-word covariance, ridge on per-word mean
activations) from a sentence's last-token residual to the hidden word's mean unit unembedding row (final basis),
fitted by scripts/al2/fit.py on ~7,200 non-benchmark words x 10 model-written sentences in five styles (scene,
description, indirect hint, two-sentence story, dialogue line) that never name the word; options chosen on 400
validation words. Candidates only LOAD the maps (.hf27/al2/). Lens: unit(JPP_l h) + lam * unit(M_l h + bias).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps  # noqa: E402,F401

BIG = "/workspace/o-lens-improvement/.hf27/al2"


def m_path(variant="all"):
    return f"{BIG}/M_{variant}.pt"


def choose_lam(ctx, maps, tag):
    base = fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"{tag} {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[al2 fit] {tag}: chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return best
