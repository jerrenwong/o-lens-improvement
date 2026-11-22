"""A-lens family (frozen for PREREG_v4 addendum E / PREREG_v5 amendment 1; user's definition, 2026-11-22).
  A-lens:  a(h) = unit( unit(E_topic h + b) + unit(E_unstated h + b') )
           the plain average of two ridge regression maps, one per question ("One-word topic:", "Unstated word:");
           no weight between them, no labels, no Jacobian.
  Rows (env ALENS_BASE):
     none -> a(h)                                  "A-lens alone"
     J    -> unit(J h)   + lam * a(h)              "J-lens + A-lens"
     JPP  -> unit(J++ h) + lam * a(h)              "J++ + A-lens"
     J0 / JPP0 -> unit(J h) / unit(J++ h)          references in the same code path (lam = 0)
  lam chosen on the model's FIT split: argmax filtered pass@1 + pass@10 over GRID, ties -> smaller lam.
Maps: .hf27/ask3/E_<tag>_t3.pt and .hf27/ask5/E_<tag>_s5.pt by model width; override with ALENS_E / ALENS_E2."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import emap, excl_mask, fmt, ranks, score  # noqa: E402
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import choose, unit  # noqa: E402

R = "/workspace/o-lens-improvement"
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
TAG = {2048: "2b", 4096: "9b", 5120: "27b"}


class A:
    def __init__(self, base, E, E2, lam):
        self.base, self.E, self.E2, self.lam = base, E, E2, lam

    def transport(self, h, layer, resid_all=None):
        if self.base is not None and self.lam == 0:
            return unit(h @ self.base[layer].T.to(h))
        a = unit(unit(emap(self.E, h, layer)) + unit(emap(self.E2, h, layer)))
        if self.base is None:
            return a
        return unit(h @ self.base[layer].T.to(h)) + self.lam * a


def build(ctx):
    mode = os.environ.get("ALENS_BASE", "JPP")
    tag = TAG.get(ctx.model.d_model)
    E = load_maps(os.environ.get("ALENS_E") or f"{R}/.hf27/ask3/E_{tag}_t3.pt", ctx.device)
    E2 = load_maps(os.environ.get("ALENS_E2") or f"{R}/.hf27/ask5/E_{tag}_s5.pt", ctx.device)
    if mode == "none":
        return A(None, E, E2, 1.0)
    base = ctx.J if mode.startswith("J") and not mode.startswith("JPP") else ctx.JPP
    if mode.endswith("0"):
        return A(base, E, E2, 0.0)
    mask = excl_mask(ctx); res = []
    for w in GRID:
        s = score(ctx, ranks(ctx, A(base, E, E2, w), mask)); res.append((w, s)); print(fmt(f"alens {mode} lam {w}", s), flush=True)
    lam = choose(res); print(f"[alens {mode}] chosen lam {lam}", flush=True)
    return A(base, E, E2, lam)
