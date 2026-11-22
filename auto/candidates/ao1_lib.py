"""ao1 shared code (not a candidate). ASK-ONLY lens: no next-token Jacobian (no J-lens, no J++ matrix) anywhere.
  v = unit(J_ask h) + lam unit(E h + b)        J_ask: Jacobian toward the asked answer (plain dict {layer: [d, d]},
                                               scripts/ja1/ja1_post.py; env AO1_J)
                                               E h + b: regression ask-map (e2 load_maps format; env AO1_E)
lam chosen on the FIT split: argmax filtered pass@1 + pass@10 over GRID, ties -> smaller lam (fm27_lib.choose).
Modes (one candidate file each):
  mix       global lam                                 perlayer  per-layer lam (coordinate ascent on FIT, 3 sweeps)
  prior     map replaced by its constant b = my        three     + mu unit(E2 h + b2) (env AO1_E2), lam fixed
  jonly     unit(J_ask h)                              maponly   unit(E h + b)
  joint     variant (b), weights from scripts/ao1/joint_fit.py (env AO1_JW)
ctx.J / ctx.JPP are never read."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import choose, excl_mask, fmt, logits, summarize, unit, NEG  # noqa: E402
from olens.readout import candidate_ranks  # noqa: E402

GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]


def load_J(ctx, var="AO1_J"):
    M = torch.load(os.environ[var], map_location="cpu", weights_only=True)
    M = {int(l): v.to(ctx.device).float() for l, v in M.items()}
    assert next(iter(M.values())).shape[0] == ctx.model.d_model, "J_ask file is for another model"
    return M


def emap(maps, h, l):
    mx, W, my = maps[l]
    return (h - mx) @ W + my


class Lens:
    """terms: list of (kind, obj, weight) with kind 'J' (h @ J^T), 'E' (map), 'b' (map's my); weight float or
    {layer: float}. Each term is unit-scaled; terms with weight 0 are skipped."""

    def __init__(self, terms):
        self.terms = terms

    def transport(self, h, layer, resid_all=None):
        v = torch.zeros_like(h)
        for kind, obj, w in self.terms:
            w = w[layer] if isinstance(w, dict) else w
            if w == 0:
                continue
            if kind == "J":
                x = h @ obj[layer].T
            elif kind == "E":
                x = emap(obj, h, layer)
            else:
                x = obj[layer][2][None].expand_as(h)
            v = v + w * unit(x)
        return v


def ranks(ctx, lens, mask):
    rk_f, rk_u = {}, {}
    for l in ctx.readout_layers:
        L = logits(ctx.model, lens.transport(ctx.fit.resid[l], l))
        rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
        rk_f[l] = candidate_ranks(L.masked_fill(mask, NEG), ctx.fit.candidates)
    return rk_f, rk_u


def score(ctx, rk):
    return summarize(ctx, *rk)


def grid_search(ctx, make, name):
    """make(w) -> Lens. Returns chosen w, {w: (rk_f, rk_u)}, [(w, summary)]."""
    mask = excl_mask(ctx)
    R, res = {}, []
    for w in GRID:
        R[w] = ranks(ctx, make(w), mask)
        s = score(ctx, R[w])
        print(fmt(f"{name} lam{w}", s), flush=True)
        res.append((w, s))
    w = choose(res)
    print(f"[ao1 {name}] chosen {w}", flush=True)
    return w, R, res


def maps_E(ctx, var="AO1_E"):
    return load_maps(os.environ[var], ctx.device)


def end_line(ctx, terms, name):
    print(fmt(f"{name} (end)", score(ctx, ranks(ctx, Lens(terms), excl_mask(ctx)))), flush=True)


def build_mix(ctx):
    J, E = load_J(ctx), maps_E(ctx)
    end_line(ctx, [("E", E, 1.0)], "map-only")
    lam, _, _ = grid_search(ctx, lambda w: Lens([("J", J, 1.0), ("E", E, w)]), "mix")
    return Lens([("J", J, 1.0), ("E", E, lam)])


def build_prior(ctx):
    J, E = load_J(ctx), maps_E(ctx)
    lam, _, _ = grid_search(ctx, lambda w: Lens([("J", J, 1.0), ("b", E, w)]), "prior")
    return Lens([("J", J, 1.0), ("b", E, lam)])


def build_perlayer(ctx):
    J, E = load_J(ctx), maps_E(ctx)
    lam, R, _ = grid_search(ctx, lambda w: Lens([("J", J, 1.0), ("E", E, w)]), "perlayer-global")
    Ls = list(ctx.readout_layers)
    cur = {l: lam for l in Ls}

    def tot(c):
        s = summarize(ctx, {l: R[c[l]][0][l] for l in Ls}, {l: R[c[l]][1][l] for l in Ls})
        return round(s["p1"] + s["p10"], 6), s

    for sweep in range(3):
        changed = False
        for l in Ls:
            best, bw = tot(cur)[0], cur[l]
            for w in GRID:
                v = tot({**cur, l: w})[0]
                if v > best:  # strict: ties keep the current (global) lam
                    best, bw = v, w
            if bw != cur[l]:
                cur[l], changed = bw, True
        print(fmt(f"perlayer sweep{sweep}", tot(cur)[1]), f"lams {cur}", flush=True)
        if not changed:
            break
    print(f"[ao1 perlayer] chosen {cur}", flush=True)
    return Lens([("J", J, 1.0), ("E", E, cur)])


def build_three(ctx):
    J, E, E2 = load_J(ctx), maps_E(ctx), maps_E(ctx, "AO1_E2")
    lam, _, _ = grid_search(ctx, lambda w: Lens([("J", J, 1.0), ("E", E, w)]), "three-step1")
    mu, _, _ = grid_search(ctx, lambda w: Lens([("J", J, 1.0), ("E", E, lam), ("E", E2, w)]), f"three lam{lam} mu")
    return Lens([("J", J, 1.0), ("E", E, lam), ("E", E2, mu)])


class Joint:
    """variant (b): v = unit(h @ J_l^T + m_l unit(E h + b)) with J_l, m_l from scripts/ao1/joint_fit.py (env AO1_JW)."""

    def __init__(self, J, E, m):
        self.J, self.E, self.m = J, E, m

    def transport(self, h, layer, resid_all=None):
        return unit(h @ self.J[layer].T + self.m[layer] * unit(emap(self.E, h, layer)))


def build_joint(ctx):
    import json
    m = {int(l): float(v) for l, v in json.load(open(os.environ["AO1_JW"])).items()}
    lens = Joint(load_J(ctx), maps_E(ctx), m)
    print(fmt(f"joint m {m}", score(ctx, ranks(ctx, lens, excl_mask(ctx)))), flush=True)
    return lens


def build_jonly(ctx):
    lens = Lens([("J", load_J(ctx), 1.0)])
    print(fmt("J_ask alone", score(ctx, ranks(ctx, lens, excl_mask(ctx)))), flush=True)
    return lens


def build_maponly(ctx):
    lens = Lens([("E", maps_E(ctx), 1.0)])
    print(fmt("map alone", score(ctx, ranks(ctx, lens, excl_mask(ctx)))), flush=True)
    return lens
