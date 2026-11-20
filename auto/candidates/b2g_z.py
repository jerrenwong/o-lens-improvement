"""Idea 14: per-token z-score readout. WikiText statistics (mean mu_t, std sd_t over ~4k positions) of the
J++ logits of unit(JPP_l h) per layer. Ideal score z_t = (logit_t - mu_t) / (sd_t + c * median(sd)).
A vector is returned by realising the top-K z scores: x = A^+ b with A = rows W_t * g of the K best tokens
(g = final-norm gain) and b = z_K - z_(K+1), so those K tokens get logits proportional to b; other tokens
get whatever x implies. c and K are chosen on the fit split; the ideal z ranking (not a vector) and the
raw ranking are printed next to it."""

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2g_common import log, unit  # noqa: E402

from olens.readout import candidate_ranks  # noqa: E402

CS = (0.0, 0.5, 1.0, 2.0, 4.0)
KS = (10, 30, 100)


def pass_from_logits(ctx, logits_by_layer, k=10):
    best = [dict() for _ in ctx.fit.candidates]
    for L in logits_by_layer.values():
        for p, d in enumerate(candidate_ranks(L, ctx.fit.candidates)):
            for kk, r in d.items():
                best[p][kk] = min(best[p].get(kk, 10 ** 9), r)
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(v) / len(v) for t, v in sorted(per.items())}
    return sum(task.values()) / len(task), task


class ZLens:
    def __init__(self, ctx, stats, c, K, W, g, excl):
        self.ctx, self.stats, self.c, self.K, self.W, self.g, self.excl = ctx, stats, c, K, W, g, excl

    def z(self, h, layer):
        mu, sd, med = self.stats[layer]
        L = self.ctx.model.unembed(unit(h @ self.ctx.JPP[layer].T)).float()
        return ((L - mu) / (sd + self.c * med)).masked_fill(self.excl, float("-inf"))

    def transport(self, h, layer, resid_all=None):
        out = []
        for i in range(0, h.shape[0], 32):
            z = self.z(h[i:i + 32], layer)
            vals, ids = z.topk(self.K + 1, dim=-1)
            b = vals[:, :-1] - vals[:, -1:]  # [n, K]
            A = self.W[ids[:, :-1]].float() * self.g  # [n, K, d]
            AAt = A @ A.transpose(1, 2)
            eps = 1e-4 * AAt.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
            alpha = torch.linalg.solve(AAt + eps * torch.eye(self.K, device=h.device), b[..., None])
            out.append((A.transpose(1, 2) @ alpha)[..., 0].to(h.device))
        return torch.cat(out, 0)


def build(ctx):
    t0 = time.time()
    model, dev = ctx.model, ctx.device
    layers = ctx.readout_layers
    W = model._lm_head.weight
    d = W.shape[1]
    g = model._final_norm(torch.ones(1, d, dtype=W.dtype, device=W.device)).float()[0]
    x = torch.randn(3, d, device=W.device)
    ref = (x * g) @ W.float().T / x.pow(2).mean(-1, keepdim=True).sqrt()
    log(f"linearity check rel err {float((model.unembed(x).float() - ref).norm() / ref.norm()):.2e}")
    # WikiText statistics
    win = ctx.wikitext("train", 32).long()
    acts = {l: [] for l in layers}
    with torch.no_grad():
        for i in range(0, win.shape[0], 8):
            rec = {}
            hs = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, inp, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[:, 4:]))(l))
                for l in layers]
            try:
                model.forward(win[i:i + 8].to(dev))
            finally:
                for hk in hs:
                    hk.remove()
            for l in layers:
                acts[l].append(rec[l].reshape(-1, d).float())
    stats = {}
    for l in layers:
        A = torch.cat(acts[l])
        s1 = s2 = 0
        for i in range(0, A.shape[0], 256):
            L = model.unembed(unit(A[i:i + 256] @ ctx.JPP[l].T)).float()
            s1 = s1 + L.sum(0)
            s2 = s2 + L.pow(2).sum(0)
        mu = s1 / A.shape[0]
        sd = (s2 / A.shape[0] - mu ** 2).clamp_min(0).sqrt()
        stats[l] = (mu, sd, sd.median())
    del acts
    log(f"wikitext stats {time.time() - t0:.0f}s, n {A.shape[0]}")
    excl = torch.zeros(W.shape[0], dtype=torch.bool, device=W.device)
    excl[ctx.fit.excluded_ids.to(W.device).long()] = True
    # ideal rankings on fit
    raw = {l: model.unembed(unit(ctx.fit.resid[l] @ ctx.JPP[l].T)).float().masked_fill(excl, float("-inf"))
           for l in layers}
    p, t = pass_from_logits(ctx, raw)
    log(f"[fit ideal] raw J++        p10 {p:.4f} | " + " ".join(f"{k[:5]} {v:.2f}" for k, v in t.items()))
    best_c = (p, None)
    for c in CS:
        zl = ZLens(ctx, stats, c, 10, W, g, excl)
        zz = {l: zl.z(ctx.fit.resid[l], l) for l in layers}
        p, t = pass_from_logits(ctx, zz)
        p1, _ = pass_from_logits(ctx, zz, 1)
        log(f"[fit ideal] z c={c:<4}      p10 {p:.4f} p1 {p1:.4f} | " + " ".join(f"{k[:5]} {v:.2f}" for k, v in t.items()))
        if best_c[1] is None or p > best_c[0]:
            best_c = (p, c)
    c = best_c[1]
    best = None
    for K in KS:
        zl = ZLens(ctx, stats, c, K, W, g, excl)
        vv = {l: model.unembed(zl.transport(ctx.fit.resid[l], l)).float().masked_fill(excl, float("-inf"))
              for l in layers}
        p, t = pass_from_logits(ctx, vv)
        p1, _ = pass_from_logits(ctx, vv, 1)
        log(f"[fit vector] c={c} K={K:<4} p10 {p:.4f} p1 {p1:.4f} | " + " ".join(f"{k[:5]} {v:.2f}" for k, v in t.items()))
        if best is None or (p, p1) > best[:2]:
            best = (p, p1, K)
    log(f"chosen c={c} K={best[2]} build {time.time() - t0:.0f}s")
    return ZLens(ctx, stats, c, best[2], W, g, excl)
