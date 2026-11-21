"""fm27 shared code (not a candidate). Why the ask-map (lens C = unit(JPP h) + lam unit(E h + b)) costs 27B its
top-1, and label-free fixes. Everything here reads ctx.fit only.

SELECTION RULE (fixed before any dev run, same for every fix and both models):
  parameter = argmax over its grid of FIT (pass@1 + pass@10), filtered scoring, computed with the exact
  transport that is then scored on dev; ties (equal to 1e-6) -> the setting closest to J++ (first in grid order).

Fixes:
  F1 rerank-only: only J++'s top-K tokens may be reordered, by zJ + mu zE (z = logit standardised over the vocab).
     Harness takes final-basis vectors, so it is realised as v = uj + delta, delta the min-norm change that sets
     the top-K logits to J++'s own sorted top-K logit values in the new order (exact on the top-K, outsiders move a
     little; the FIT print shows exact vs vector).
  F2 centred map: unit(JPP h) + lam unit(E h + b - b_prior), b_prior = my (the ridge map's mean target).
  F3 top-1 protection: C, plus the smallest push c * unit(gamma * W_t) along J++'s own filtered top-1 token t that
     makes t rank 1 (bisection); rows where C already has t at rank 1 are untouched.
  F4 lam re-chosen on FIT by the rule above.
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps, t3_path  # noqa: E402
from olens.readout import candidate_ranks, unembed_logits  # noqa: E402

LAM_C = {4096: 0.2, 5120: 0.5}  # registered C (chosen on FIT by ask3_lib.choose_lam)
TAG = {4096: "9b", 5120: "27b"}
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
KS = [20, 50, 100]
MUS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
NEG = float("-inf")


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def excl_mask(ctx):
    V = ctx.model._lm_head.weight.shape[0]
    m = torch.zeros(V, dtype=torch.bool, device=ctx.model._lm_head.weight.device)
    m[ctx.fit.excluded_ids.to(m.device)] = True
    return m


def logits(model, v):
    return unembed_logits(model, v, None, chunk_size=16)


def head_gamma(model):
    """gamma such that unembed(v) = W (gamma * v / rms(v)) (same measurement as olens.readout.final_linear)."""
    norm, head = model._final_norm, model._lm_head
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    p = next(norm.parameters())
    ones = torch.ones(1, head.weight.shape[1], dtype=torch.float32, device=p.device)
    return (norm(ones).float()[0] * (1.0 + eps) ** 0.5).to(head.weight.device)


# ------------------------------------------------------------------ fit scoring


def pass_from_best(ctx, best, k):
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(v) / len(v) for t, v in sorted(per.items())}
    return sum(task.values()) / len(task), task


def best_of(per_layer):
    """per_layer: {layer: [ {inter: rank} per row ]} -> [ {inter: best rank} per row ]."""
    layers = list(per_layer)
    n = len(per_layer[layers[0]])
    out = [dict() for _ in range(n)]
    for l in layers:
        for p, d in enumerate(per_layer[l]):
            for k, r in d.items():
                out[p][k] = min(out[p].get(k, 10 ** 9), r)
    return out


def summarize(ctx, rk_f, rk_u):
    bf, bu = best_of(rk_f), best_of(rk_u)
    p10, task = pass_from_best(ctx, bf, 10)
    p1, _ = pass_from_best(ctx, bf, 1)
    p10u, _ = pass_from_best(ctx, bu, 10)
    p1u, _ = pass_from_best(ctx, bu, 1)
    return dict(p1=p1, p10=p10, p1u=p1u, p10u=p10u, task=task)


def fmt(name, s):
    tasks = " ".join(f"{t[:5]} {v:.2f}" for t, v in s["task"].items())
    return (f"{name:28s} p10 {s['p10']:.4f} p1 {s['p1']:.4f} | unf p10 {s['p10u']:.4f} p1 {s['p1u']:.4f} | "
            f"{tasks}")


def fit_eval_lens(ctx, lens, mask):
    rk_f, rk_u = {}, {}
    for l in ctx.readout_layers:
        v = lens.transport(ctx.fit.resid[l], l, ctx.fit.resid)
        L = logits(ctx.model, v)
        rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
        rk_f[l] = candidate_ranks(L.masked_fill(mask, NEG), ctx.fit.candidates)
    return summarize(ctx, rk_f, rk_u)


def choose(results):
    """results: list of (param, summary) in grid order (closest to J++ first). Rule: max p1 + p10 (filtered)."""
    best = max(range(len(results)), key=lambda i: (round(results[i][1]["p1"] + results[i][1]["p10"], 6), -i))
    return results[best][0]


# ------------------------------------------------------------------ lenses


class Base:
    def __init__(self, ctx, lam=0.0, centred=False):
        self.JPP = ctx.JPP
        self.maps = load_maps(t3_path(ctx), ctx.device)
        self.lam, self.centred = lam, centred

    def parts(self, h, l):
        mx, W, my = self.maps[l]
        e = (h - mx) @ W
        if not self.centred:
            e = e + my
        return unit(h @ self.JPP[l].T), unit(e)

    def transport(self, h, layer, resid_all=None):
        uj, ue = self.parts(h, layer)
        return uj + self.lam * ue


class F1Lens(Base):
    """Rerank only J++'s filtered top-K by zJ + mu zE, as a final-basis vector (min-norm logit edit)."""

    def __init__(self, ctx, K, mu):
        super().__init__(ctx)
        self.K, self.mu, self.model = K, mu, ctx.model
        self.mask = excl_mask(ctx)
        self.gamma = head_gamma(ctx.model)
        self.Wh = ctx.model._lm_head.weight

    @staticmethod
    def z(L, mask):
        keep = ~mask
        x = L[:, keep]
        return (L - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)

    def order(self, uj, ue):
        Lj, Le = logits(self.model, uj), logits(self.model, ue)
        Ljm = Lj.masked_fill(self.mask, NEG)
        top = Ljm.topk(self.K, dim=-1).indices  # [n, K], J++ order
        s = self.z(Lj, self.mask).gather(1, top) + self.mu * self.z(Le, self.mask).gather(1, top)
        new = top.gather(1, s.argsort(dim=-1, descending=True))  # top-K ids in F1 order
        return top, new

    def transport(self, h, layer, resid_all=None):
        uj, ue = self.parts(h, layer)
        top, new = self.order(uj, ue)
        out = []
        for i in range(0, uj.shape[0], 16):
            A = self.Wh[new[i:i + 16]].float() * self.gamma  # [b, K, d], rows in F1 order
            cur = torch.einsum("bkd,bd->bk", A, uj[i:i + 16])
            tgt = cur.sort(dim=-1, descending=True).values  # J++'s own top-K logit values, assigned in F1 order
            G = A @ A.transpose(1, 2) + 1e-4 * torch.eye(A.shape[1], device=A.device)
            a = torch.linalg.solve(G, (tgt - cur).unsqueeze(-1))
            out.append(uj[i:i + 16] + (A.transpose(1, 2) @ a).squeeze(-1))
        return torch.cat(out, 0)


class F3Lens(Base):
    """C with J++'s filtered top-1 pushed back to rank 1 by the smallest push along unit(gamma * W_t)."""

    def __init__(self, ctx, lam):
        super().__init__(ctx, lam)
        self.model, self.mask = ctx.model, excl_mask(ctx)
        self.gamma = head_gamma(ctx.model)
        self.Wh = ctx.model._lm_head.weight
        self.fallback = 0

    def ok(self, v, t):
        L = logits(self.model, v).masked_fill(self.mask, NEG)
        lt = L.gather(1, t[:, None])
        return (L > lt).sum(-1) == 0

    def transport(self, h, layer, resid_all=None):
        uj, ue = self.parts(h, layer)
        t = logits(self.model, uj).masked_fill(self.mask, NEG).argmax(-1)
        v = uj + self.lam * ue
        bad = (~self.ok(v, t)).nonzero().flatten()
        if len(bad) == 0:
            return v
        d = unit(self.Wh[t[bad]].float() * self.gamma)
        vb, tb = v[bad], t[bad]
        lo = torch.zeros(len(bad), device=v.device)
        hi = torch.full((len(bad),), 0.25, device=v.device)
        for _ in range(14):  # grow hi until every row is fixed
            good = self.ok(vb + hi[:, None] * d, tb)
            if good.all():
                break
            hi = torch.where(good, hi, hi * 2)
        good = self.ok(vb + hi[:, None] * d, tb)
        self.fallback += int((~good).sum())
        for _ in range(16):
            mid = (lo + hi) / 2
            g = self.ok(vb + mid[:, None] * d, tb)
            hi, lo = torch.where(g, mid, hi), torch.where(g, lo, mid)
        c = torch.where(good, hi, torch.zeros_like(hi))
        v = v.clone()
        v[bad] = vb + c[:, None] * d
        return v
