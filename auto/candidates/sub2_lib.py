"""Shared code for sub2_* candidates (not a candidate itself).

M_l is a LEARNED MAP, not a Jacobian: al2's noise-aware ridge (scripts/al2/fit.py recipe) refitted by
scripts/sub2/fit2.py on SUBSETS of al2's model-written train sentences, selected by whether the 9B model itself can
name the hidden word from the sentence (rank <= 10) and from its word-shuffled version (scripts/sub2/label.py):
ALL, SUB (inferable, not given away), INF (inferable), RSUB0 / RINF0 (random sentences matched per word to SUB / INF).
Candidates only LOAD the maps (.hf27/sub2/). Lens: unit(JPP_l h) + lam * unit(M_l h + bias), lam chosen on the FIT
split exactly as al2_T (al2_lib.choose_lam).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al2_lib import ELens, choose_lam, fit_line, load_maps  # noqa: E402,F401

BIG = "/workspace/o-lens-improvement/.hf27/sub2"


def build_sub2(ctx, variant):
    maps = load_maps(f"{BIG}/M_{variant}.pt", ctx.device)
    lam = choose_lam(ctx, maps, f"M_{variant} mix")
    fit_line("M alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
