"""s1: stack = a saved J++ lens file (env S1_LENS, e.g. label-free weights) + lam * ask-map E (e2_lib).
lam chosen on the model's FIT split over e2_lib.LAMS (ties -> smaller), exactly as e2_A does. 9B only."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps, map_path  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402


def build(ctx):
    L = load_lens_file(os.environ["S1_LENS"], hf_model_name="Qwen/Qwen3.5-9B")
    J = {int(l): v.to(ctx.device).float() for l, v in L.jacobians_L_dict_FN.items()}
    maps = load_maps(map_path(ctx), ctx.device)
    base = fit_line("base", ctx, ELens(J, maps, "mix", 0.0))
    scores = {lam: fit_line(f"mix {lam}", ctx, ELens(J, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[s1 fit] chosen lam {best} (fit {scores[best]:.4f} vs base {base:.4f})", flush=True)
    return ELens(J, maps, "mix", best)
