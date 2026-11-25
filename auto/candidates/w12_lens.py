"""w12 (Experiment 5B): Jacobian-lens maps trained to match the model's output probabilities (scripts/w12/w12_train.py).
  trained map:  t(h) = h @ M_l^T          (M_l [d,d] per readout layer; file W12_MAP = {layer: {"M", "tau"}})
Rows (env W12_BASE):
  none  -> t(h)                                   the trained map alone
  J     -> unit(J h)   + beta * unit(t(h))        added to the frozen J-lens
  JPP   -> unit(J++ h) + beta * unit(t(h))        added to the frozen J++ (the evaluator's J++ matrix)
  J0 / JPP0 -> unit(J h) / unit(J++ h)            references in the same code path
W12_ALENS=1: + lam * a(h), a(h) = the frozen A-lens (CE 98k maps, w9 format; env ALENS_E / ALENS_E2 override) as
  auto/candidates/alens.py + w9_alens.py:  v = unit(lens(h)) + lam * unit(unit(E h) + unit(E2 h)).
beta, then lam (with beta fixed), each chosen on the FIT split under the evaluator's --filter level exactly as alens.py
/ w1_alens.py: argmax filtered pass@1 + pass@10 over GRID, ties -> smaller value; ranks over token groups at f3/f4.
W12_BETA / W12_LAM fix a value (no search)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import NEG, choose, excl_mask, fmt, logits, summarize, unit  # noqa: E402
from olens.readout import candidate_ranks, group_candidate_ranks  # noqa: E402

R = "/workspace/o-lens-improvement"
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
TAG = {2048: "2b", 4096: "9b", 5120: "27b"}


def load_map(path, dev):
    m = torch.load(path, map_location="cpu")
    return {int(l): v["M"].to(dev).float() for l, v in m.items()}


def load_maps_w9(path, dev):
    m = torch.load(path, map_location="cpu")
    out = {}
    for l, v in m.items():
        assert v.get("W1") is None, "w12 uses the linear A-lens maps only"
        out[l] = (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
    return out


def emap(m, h, l):
    mx, W, my = m[l]
    return (h - mx) @ W + my


class Lens:
    def __init__(self, base, M, beta, A=None, lam=0.0):
        self.base, self.M, self.beta, self.A, self.lam = base, M, beta, A, lam

    def core(self, h, layer):
        if self.base is None:
            return h @ self.M[layer].T.to(h)
        b = unit(h @ self.base[layer].T.to(h))
        if self.beta == 0 or self.M is None:
            return b
        return b + self.beta * unit(h @ self.M[layer].T.to(h))

    def transport(self, h, layer, resid_all=None):
        v = self.core(h, layer)
        if self.A is None or self.lam == 0:
            return v
        E, E2 = self.A
        return unit(v) + self.lam * unit(unit(emap(E, h, layer)) + unit(emap(E2, h, layer)))


def ranks(ctx, lens, mask):
    gid = getattr(ctx.fit, "group_ids", None)
    rk_f, rk_u = {}, {}
    for l in ctx.readout_layers:
        L = logits(ctx.model, lens.transport(ctx.fit.resid[l], l))
        rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
        Lm = L.masked_fill(mask, NEG)
        rk_f[l] = candidate_ranks(Lm, ctx.fit.candidates) if gid is None else group_candidate_ranks(Lm, ctx.fit.candidates, gid)
    return rk_f, rk_u


def search(ctx, make, name, fixed):
    if fixed is not None:
        return float(fixed)
    mask = excl_mask(ctx)
    res = []
    for w in GRID:
        s = summarize(ctx, *ranks(ctx, make(w), mask)); res.append((w, s)); print(fmt(f"{name} {w}", s), flush=True)
    w = choose(res)
    print(f"[w12 {name}] chosen {w}", flush=True)
    return w


def build(ctx):
    mode = os.environ.get("W12_BASE", "none")
    print(f"[w12] filter {getattr(ctx.fit, 'filter_level', 'f1')}, base {mode}, map {os.environ.get('W12_MAP')}, "
          f"alens {os.environ.get('W12_ALENS', '0')}", flush=True)
    M = load_map(os.environ["W12_MAP"], ctx.device) if not mode.endswith("0") else None
    if mode == "none":
        base, beta = None, 1.0
    else:
        base = ctx.J if mode in ("J", "J0") else ctx.JPP
        beta = 0.0 if mode.endswith("0") else search(ctx, lambda w: Lens(base, M, w), f"beta {mode}",
                                                       os.environ.get("W12_BETA"))
    if os.environ.get("W12_ALENS", "0") != "1":
        lens = Lens(base, M, beta)
        print(fmt(f"FIT final ({mode}, beta {beta})", summarize(ctx, *ranks(ctx, lens, excl_mask(ctx)))), flush=True)
        return lens
    tag = TAG[ctx.model.d_model]
    A = (load_maps_w9(os.environ.get("ALENS_E") or f"{R}/.hf27/w9/maps/E_{tag}_topic_lin_ce_98k.pt", ctx.device),
         load_maps_w9(os.environ.get("ALENS_E2") or f"{R}/.hf27/w9/maps/E_{tag}_spec_lin_ce_98k.pt", ctx.device))
    lam = search(ctx, lambda w: Lens(base, M, beta, A, w), f"lam (beta {beta})", os.environ.get("W12_LAM"))
    lens = Lens(base, M, beta, A, lam)
    print(fmt(f"FIT final ({mode}, beta {beta}, lam {lam})", summarize(ctx, *ranks(ctx, lens, excl_mask(ctx)))), flush=True)
    return lens
