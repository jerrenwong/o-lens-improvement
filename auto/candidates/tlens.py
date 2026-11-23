"""T-lens family (PREREG_v4 Addendum F, text-only control of the A-lens; mirrors auto/candidates/alens.py).
  T-lens:  t(f) = unit( unit(T_topic f + b) + unit(T_unstated f + b') )
           f = mean INPUT embedding of the row's prompt tokens [0 .. readout position] (no forward pass);
           T_topic / T_unstated = the A-lens ridge recipe (same passages, targets, grid, validation and junk rules)
           with f as the regression input; one map per question, used at every readout layer (scripts/tl1/tl1_fit.py).
  Rows (env TLENS_BASE):
     none -> t(f)                                  "T-lens alone"
     J    -> unit(J h)   + lam * t(f)              "J-lens + T-lens"
     JPP  -> unit(J++ h) + lam * t(f)              "J++ + T-lens"
     J0 / JPP0 -> unit(J h) / unit(J++ h)          references in the same code path (lam = 0)
  lam chosen on the model's FIT split: argmax filtered pass@1 + pass@10 over GRID, ties -> smaller lam (as alens.py).
Maps: env TLENS_E (topic) and TLENS_E2 (unstated), both required (.hf27/tl1/T_<tag>_t3.pt, T_<tag>_s5.pt).
Needs a scorer that supplies the feature: auto/evaluate_t.py / auto/evaluate_t_v4.py (ctx.fit.tfeat for the FIT rows,
transport(..., tfeat=...) for the scored rows)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import emap, excl_mask, fmt, ranks, score  # noqa: E402
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import choose, unit  # noqa: E402

GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


class T:
    def __init__(self, base, E, E2, lam, feat):
        self.base, self.E, self.E2, self.lam, self.feat = base, E, E2, lam, feat

    def transport(self, h, layer, resid_all=None, tfeat=None):
        if self.base is not None and self.lam == 0:
            return unit(h @ self.base[layer].T.to(h))
        f = self.feat if tfeat is None else tfeat  # FIT rows during build; scored rows from the scorer
        assert f.shape[0] == h.shape[0], f"feature rows {f.shape[0]} != activation rows {h.shape[0]}"
        f = f.to(h)
        t = unit(unit(emap(self.E, f, layer)) + unit(emap(self.E2, f, layer)))
        if self.base is None:
            return t
        return unit(h @ self.base[layer].T.to(h)) + self.lam * t


def build(ctx):
    mode = os.environ.get("TLENS_BASE", "JPP")
    E = load_maps(os.environ["TLENS_E"], ctx.device)
    E2 = load_maps(os.environ["TLENS_E2"], ctx.device)
    assert set(ctx.readout_layers) <= set(E) and set(ctx.readout_layers) <= set(E2), "maps lack a readout layer"
    feat = ctx.fit.tfeat
    if mode == "none":
        return T(None, E, E2, 1.0, feat)
    base = ctx.J if mode.startswith("J") and not mode.startswith("JPP") else ctx.JPP
    if mode.endswith("0"):
        return T(base, E, E2, 0.0, feat)
    mask = excl_mask(ctx); res = []
    for w in GRID:
        s = score(ctx, ranks(ctx, T(base, E, E2, w, feat), mask)); res.append((w, s)); print(fmt(f"tlens {mode} lam {w}", s), flush=True)
    lam = choose(res); print(f"[tlens {mode}] chosen lam {lam}", flush=True)
    return T(base, E, E2, lam, feat)
