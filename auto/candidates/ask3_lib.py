"""ask3 shared code (not a candidate). E_t3: the e2/m27 ask-map recipe (LEARNED ridge map, not a Jacobian) refit on
27B topic targets elicited in a format whose answer position is a word, not a blank line
(scripts/ask3/ask3_collect.py, ask3_fit.py). Maps: .hf27/ask3/E_27b_t3.pt, E_27b_t3_shuf.pt.
lam grid extended to 2.0 (ties -> smaller)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import ELens, fit_line, load_maps  # noqa: E402,F401

R = "/workspace/o-lens-improvement"
TAGS = {4096: "9b", 5120: "27b"}
LAMS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def t3_path(ctx, shuf=False):
    return f"{R}/.hf27/ask3/E_{TAGS[ctx.model.d_model]}_t3{'_shuf' if shuf else ''}.pt"


def choose_lam(ctx, maps):
    fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"t3 mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[ask3 fit] chosen lam {best} fit {scores[best]:.4f}", flush=True)
    return best
