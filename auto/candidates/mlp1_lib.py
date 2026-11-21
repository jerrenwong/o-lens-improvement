"""Shared code for mlp1_* candidates (not a candidate itself).

f_l is a LEARNED MAP, not a Jacobian, fitted by scripts/mlp1/mlp1_train.py on the al2 data (model-written sentences
that imply but never state a word; ~7,100 training words, 5 styles): from the last-token residual to a vector decoded
through the frozen final norm gain and unembedding. Kinds (z = (h - mx) @ P, P = al2 M3 whitener; u = z / zstd):
  R: ridge M3 (al2 recipe): z W + b
  B: linear, initialised at ridge, trained with full-vocabulary cross-entropy, early-stopped on validation words
  C: z W + b + s_f * Bo relu(A u + a)   (residual MLP, ridge-initialised skip)
  D: s_f * (b + Bo relu(A u + a))       (MLP from scratch)
Candidates only LOAD the maps (.hf27/mlp1/F_<group>.pt). Lens: unit(JPP_l h) + lam * unit(f_l(h)), lam chosen on FIT.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, fit_line  # noqa: E402,F401
from b2d_common import unit  # noqa: E402

BIG = "/workspace/o-lens-improvement/.hf27/mlp1"


def load_f(group, dev):
    p = torch.load(f"{BIG}/F_{group}.pt", map_location="cpu", weights_only=False)
    return {int(l): dict(kind=v["kind"], s_f=v["s_f"],
                         st={k: t.to(dev).float() for k, t in v["state"].items() if torch.is_tensor(t)})
            for l, v in p.items()}


def apply_f(fl, h):
    st, kind = fl["st"], fl["kind"]
    z = (h.float() - st["mx"]) @ st["P"]
    if kind == "P":
        return z @ st["W"] + st["b0"] + fl["s_f"] * st["c"]
    if kind == "D":
        return fl["s_f"] * (st["b"] + torch.relu((z / st["zstd"]) @ st["A.weight"].T + st["A.bias"]) @ st["Bo.weight"].T)
    out = z @ st["W"] + st["b"]
    if kind == "Q":
        out = out + fl["s_f"] * st["c"]
    if kind in ("C", "L"):
        act = torch.relu if kind == "C" else (lambda t: t)
        out = out + fl["s_f"] * act((z / st["zstd"]) @ st["A.weight"].T + st["A.bias"]) @ st["Bo.weight"].T
    return out


class FLens:
    """mode 'mix': unit(JPP h) + lam unit(f(h)); 'alone': f(h)."""

    def __init__(self, JPP, F, mode, lam=None):
        self.JPP, self.F, self.mode, self.lam = JPP, F, mode, lam

    def transport(self, h, layer, resid_all=None):
        f = apply_f(self.F[layer], h)
        if self.mode == "alone":
            return f
        return unit(h @ self.JPP[layer].T) + self.lam * unit(f)


def build_group(ctx, group, tag):
    F = load_f(group, ctx.device)
    base = fit_line("jpp", ctx, FLens(ctx.JPP, F, "mix", 0.0))
    scores = {lam: fit_line(f"{tag} {lam}", ctx, FLens(ctx.JPP, F, "mix", lam)) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(scores[z], 6), -z))
    print(f"[mlp1 fit] {tag}: chosen lam {lam} (fit {scores[lam]:.4f} vs J++ {base:.4f})", flush=True)
    fit_line(f"{tag} alone", ctx, FLens(ctx.JPP, F, "alone"))
    return FLens(ctx.JPP, F, "mix", lam)
