"""ej1: the elicited J-lens (scripts/ej1/ej1_fit.py). E = averaged Jacobian toward the model's states after an inline
elicitation suffix. env:
  EJ1_E      path to a plain dict {layer: [d, d]} (several paths joined by ',' -> E = mean of the unit-free matrices
             after each is scaled to unit Frobenius norm per layer: the pooled-suffix lens)
  EJ1_PRIOR  optional 'path:key' of prior.pt: e(h) = unit(unit(E h) - m_l), m_l = mean unit(E h) over 500 unrelated
             activations (prior-removed variant); default e(h) = unit(E h)
  EJ1_MODE   alone -> e(h) | J -> unit(J h) + beta e(h) | JPP -> unit(J++ h) + beta e(h)
             JPPA  -> unit(J++ h) + beta e(h) + lam a(h), a = the A-lens with the w9 CE 98k maps (w9_alens conventions;
             ALENS_E / ALENS_E2 default .hf27/w9/maps/E_<tag>_{topic,spec}_lin_ce_98k.pt)
             J0 / JPP0 -> the base alone in this code path
beta (and lam) chosen on the FIT split with alens.GRID and fm27_lib.choose (max filtered p1 + p10; ties -> smaller),
FIT ranks over the evaluator's filter groups as in w3_f.py / w1_alens.py."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alens import GRID, TAG  # noqa: E402
from ao1_lib import excl_mask, fmt, score  # noqa: E402
from fm27_lib import NEG, choose, logits, unit  # noqa: E402
from olens.readout import candidate_ranks, group_candidate_ranks  # noqa: E402

R = "/workspace/o-lens-improvement"


def ranks(ctx, lens, mask):
    gid = getattr(ctx.fit, "group_ids", None)
    rk_f, rk_u = {}, {}
    for l in ctx.readout_layers:
        L = logits(ctx.model, lens.transport(ctx.fit.resid[l], l))
        rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
        Lm = L.masked_fill(mask, NEG)
        rk_f[l] = (candidate_ranks(Lm, ctx.fit.candidates) if gid is None
                   else group_candidate_ranks(Lm, ctx.fit.candidates, gid))
    return rk_f, rk_u


def load_E(ctx):
    paths = [p for p in os.environ["EJ1_E"].split(",") if p]
    Es = []
    for p in paths:
        M = torch.load(p if p.startswith("/") else f"{R}/{p}", map_location="cpu", weights_only=True)
        M = {int(l): v.to(ctx.device).float() for l, v in M.items()}
        assert next(iter(M.values())).shape[0] == ctx.model.d_model, f"{p} is for another model"
        Es.append(M)
    if len(Es) == 1:
        return Es[0]
    return {l: sum(E[l] / E[l].norm() for E in Es) / len(Es) for l in Es[0]}


def load_prior(ctx):
    s = os.environ.get("EJ1_PRIOR")
    if not s:
        return None
    p, key = s.rsplit(":", 1)
    P = torch.load(p if p.startswith("/") else f"{R}/{p}", map_location="cpu", weights_only=True)[key]
    return {int(l): v.to(ctx.device).float() for l, v in P.items()}


class EJ:
    def __init__(self, E, prior=None, base=None, beta=1.0, A=None, lam=0.0):
        self.E, self.prior, self.base, self.beta, self.A, self.lam = E, prior, base, beta, A, lam

    def e(self, h, layer):
        v = unit(h @ self.E[layer].T)
        return unit(v - self.prior[layer]) if self.prior is not None else v

    def a(self, h, layer):
        import w9_alens
        E1, E2 = self.A
        return unit(unit(w9_alens.emap_w9(E1, h, layer)) + unit(w9_alens.emap_w9(E2, h, layer)))

    def transport(self, h, layer, resid_all=None):
        if self.base is None:
            return self.e(h, layer)
        v = unit(h @ self.base[layer].T.to(h))
        if self.beta:
            v = v + self.beta * self.e(h, layer)
        if self.A is not None and self.lam:
            v = v + self.lam * self.a(h, layer)
        return v


def build(ctx):
    mode = os.environ.get("EJ1_MODE", "alone")
    mask = excl_mask(ctx)
    E = load_E(ctx) if mode not in ("J0", "JPP0") else None
    prior = load_prior(ctx)
    if mode == "alone":
        lens = EJ(E, prior)
        print(fmt("ej1 alone FIT", score(ctx, ranks(ctx, lens, mask))), flush=True)
        return lens
    base = ctx.J if mode in ("J", "J0") else ctx.JPP
    if mode.endswith("0"):
        return EJ(None, None, base, 0.0)
    A = None
    if mode == "JPPA":
        import w9_alens
        tag = TAG[ctx.model.d_model]
        A = (w9_alens.load_maps_w9(f"{R}/" + os.environ.get("ALENS_E", f".hf27/w9/maps/E_{tag}_topic_lin_ce_98k.pt"), ctx.device),
             w9_alens.load_maps_w9(f"{R}/" + os.environ.get("ALENS_E2", f".hf27/w9/maps/E_{tag}_spec_lin_ce_98k.pt"), ctx.device))
    res = []
    lams = GRID if A is not None else [0.0]
    for lam in lams:
        for b in GRID:
            s = score(ctx, ranks(ctx, EJ(E, prior, base, b, A, lam), mask))
            res.append(((b, lam), s))
            print(fmt(f"ej1 {mode} beta {b} lam {lam}", s), flush=True)
    # ties -> earlier in grid order (smaller lam, then smaller beta)
    b, lam = choose(res)
    print(f"[ej1 {mode}] chosen beta {b} lam {lam}", flush=True)
    return EJ(E, prior, base, b, A, lam)
