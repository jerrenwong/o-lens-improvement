"""Shared code for b2f_* candidates (not a candidate itself).

Idea 9  future-bag ridge lens: closed-form ridge from h_l[t] (WikiText) to the IDF-weighted mean unembedding
        row of the content tokens at t+2..t+H (next token excluded). Lens: unit(JPP h) + b * unit(A (h - mu) / gamma).
Idea 11 far-target Jacobian: random-probe VJP estimate of sum_{t far after s} d h_final[t] / d h_l[s]
        (cotangent only at the last positions of short windows, sources >= KMIN tokens earlier).
Idea 13 interventional lens: write h into WikiText contexts at layer l, mean change of later final states
        vs writing the WikiText mean activation.
Fit helpers print per-task pass@k on ctx.fit (the labelled FIT split) from build().
"""

import math
import time
from collections import defaultdict

import torch

from olens.readout import candidate_ranks, unembed_logits


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


class _Stop(Exception):
    pass


def record(model, ids, layers, stop_after=None):
    """Forward (no grad) recording block outputs at `layers`; stops after block `stop_after`."""
    rec = {}
    stop_after = max(layers) if stop_after is None else stop_after

    def mk(l):
        def hook(m, i, o):
            rec[l] = o if torch.is_tensor(o) else o[0]
            if l == stop_after:
                raise _Stop
        return hook

    hs = [model.layers[l].register_forward_hook(mk(l)) for l in sorted(set(layers) | {stop_after})]
    try:
        model.forward(ids)
    except _Stop:
        pass
    finally:
        for x in hs:
            x.remove()
    return rec


def final_gamma(model):
    norm = model._final_norm
    d = model.d_model
    w = next(norm.parameters())
    ones = torch.ones(1, d, dtype=w.dtype, device=w.device)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    return (norm(ones).float()[0] * (1.0 + eps) ** 0.5)


# ---------------------------------------------------------------- fit scoring


def fit_ranks(ctx, vecs):
    best = [dict() for _ in ctx.fit.candidates]
    for layer, v in vecs.items():
        logits = unembed_logits(ctx.model, v, ctx.fit.excluded_ids, chunk_size=16)
        for p, d in enumerate(candidate_ranks(logits, ctx.fit.candidates)):
            for k, r in d.items():
                best[p][k] = min(best[p].get(k, 10 ** 9), r)
    return best


def fit_pass(ctx, vecs, k=10):
    best = fit_ranks(ctx, vecs)
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(v) / len(v) for t, v in sorted(per.items())}
    return sum(task.values()) / len(task), task


def report(ctx, name, vecs):
    p10, t10 = fit_pass(ctx, vecs, 10)
    p1, _ = fit_pass(ctx, vecs, 1)
    tasks = " ".join(f"{t[:5]} {v:.2f}" for t, v in t10.items())
    print(f"[fit] {name:34s} p10 {p10:.4f} p1 {p1:.4f} | {tasks}", flush=True)
    return p10, p1


# ---------------------------------------------------------------- lenses


class LinLens:
    """unit(JPP_l h) + lam * unit(M_l (h - mu_l) / gamma); lam=None -> the expert alone."""

    def __init__(self, JPP, M, mu, gamma, lam=None):
        self.JPP, self.M, self.mu, self.gamma, self.lam = JPP, M, mu, gamma, lam

    def expert(self, h, layer):
        x = (h - self.mu[layer]) @ self.M[layer]
        return x / self.gamma if self.gamma is not None else x

    def transport(self, h, layer, resid_all=None):
        x = self.expert(h, layer)
        if self.lam is None:
            return x
        return unit(h @ self.JPP[layer].T) + self.lam * unit(x)


# ---------------------------------------------------------------- idea 9: future-bag ridge


def _first_occ(ids):
    """[B, T] -> [B, T] index of the first occurrence of ids[b, j] within the window."""
    eq = ids[:, :, None] == ids[:, None, :]  # [B, j, i]
    T = ids.shape[1]
    ar = torch.arange(T, device=ids.device)
    return torch.where(eq, ar[None, None, :], T).min(-1).values


def future_targets(ids, W, wv, H, t0, t1, novel):
    """Y [B, t1-t0, d] fp32: weighted mean of W rows of tokens at t+2..t+H; valid mask [B, t1-t0]."""
    B, T = ids.shape
    wt = wv[ids]  # [B, T]
    if novel:
        f = _first_occ(ids)
    num = torch.zeros(B, t1 - t0, W.shape[1], device=ids.device)
    den = torch.zeros(B, t1 - t0, device=ids.device)
    tt = torch.arange(t0, t1, device=ids.device)
    for k in range(2, H + 1):
        j = tt + k
        w = wt[:, j]
        if novel:
            w = w * (f[:, j] >= (tt + 2)[None]).float()
        num += w[..., None] * W[ids[:, j]].float()
        den += w
    return num / den.clamp_min(1e-6)[..., None], den > 0


def fit_fbag(ctx, n_win=4096, H=32, batch=32, alphas=(1e-4, 1e-3, 1e-2, 1e-1), n_val=192, targets=("all", "novel")):
    """Returns {target: {alpha: {layer: A [d, d]}}}, mu {layer: [d]}, and prints label-free sanity."""
    model, dev, L = ctx.model, ctx.device, ctx.readout_layers
    t_start = time.time()
    W = model._lm_head.weight.detach()  # [V, d] bf16
    V, d = W.shape
    excl = ctx.fit.excluded_ids.to(dev).long()
    win = ctx.wikitext("train", 81920)[:n_win].long()
    val = ctx.wikitext("validation", 512)[:n_val].long()
    # IDF over training windows
    df = torch.zeros(V, device=dev)
    for i in range(0, n_win, 512):
        x = win[i:i + 512].to(dev)
        pres = torch.zeros(x.shape[0], V, device=dev, dtype=torch.bool)
        pres.scatter_(1, x, True)
        df += pres.float().sum(0)
    wv = torch.log(n_win / (1.0 + df)).clamp_min(0.0)
    wv[excl] = 0.0
    T = win.shape[1]
    t0, t1 = 16, T - H  # positions with a full future window
    n = 0
    shift = {}
    Sx = {l: torch.zeros(d, dtype=torch.float64, device=dev) for l in L}
    XtX = {l: torch.zeros(d, d, dtype=torch.float64, device=dev) for l in L}
    Sy = {tg: torch.zeros(d, dtype=torch.float64, device=dev) for tg in targets}
    XtY = {tg: {l: torch.zeros(d, d, dtype=torch.float64, device=dev) for l in L} for tg in targets}
    for i in range(0, n_win, batch):
        ids = win[i:i + batch].to(dev)
        rec = record(model, ids, L)
        Ys = {}
        for tg in targets:
            Y, ok = future_targets(ids, W, wv, H, t0, t1, tg == "novel")
            Ys[tg] = Y.reshape(-1, d)
        okf = ok.reshape(-1)  # validity of 'novel' is a subset; use the last (strictest) mask
        for tg in targets:
            Ys[tg] = Ys[tg][okf]
        nb = int(okf.sum())
        n += nb
        for tg in targets:
            Sy[tg] += Ys[tg].sum(0).double()
        for l in L:
            X = rec[l][:, t0:t1].reshape(-1, d).float()[okf]
            if l not in shift:  # fixed shift from the first batch: avoids fp cancellation when centring
                shift[l] = X.mean(0)
            X = X - shift[l]
            Sx[l] += X.sum(0).double()
            XtX[l] += (X.T @ X).double()
            for tg in targets:
                XtY[tg][l] += (X.T @ Ys[tg]).double()
    print(f"[fbag] accumulated {n} positions from {n_win} windows in {time.time() - t_start:.0f}s", flush=True)
    mu = {l: (Sx[l] / n) for l in L}  # mean of shifted X
    muy = {tg: Sy[tg] / n for tg in targets}
    # held-out data
    vX = {l: [] for l in L}
    vY = {tg: [] for tg in targets}
    vB = []  # future bag token ids (for precision) [P, H-1], -1 where weight 0
    for i in range(0, n_val, batch):
        ids = val[i:i + batch].to(dev)
        rec = record(model, ids, L)
        for tg in targets:
            Y, ok = future_targets(ids, W, wv, H, t0, t1, tg == "novel")
            vY[tg].append(Y.reshape(-1, d)[::8])
        for l in L:
            vX[l].append(rec[l][:, t0:t1].reshape(-1, d).float()[::8])
        tt = torch.arange(t0, t1, device=dev)
        J = tt[:, None] + torch.arange(2, H + 1, device=dev)[None]  # [P, H-1]
        bag = ids[:, J]  # [B, P, H-1]
        bag = torch.where(wv[bag] > 0, bag, torch.full_like(bag, -1))
        vB.append(bag.reshape(-1, H - 1)[::8])
    vX = {l: torch.cat(v) for l, v in vX.items()}
    vY = {tg: torch.cat(v) for tg, v in vY.items()}
    vB = torch.cat(vB)
    novel_mask = None
    print(f"[fbag] held-out positions {vB.shape[0]}", flush=True)

    def prec10(P):
        hits = []
        for c in range(0, P.shape[0], 512):
            logits = (P[c:c + 512].to(W.dtype) @ W.T).float()
            logits[:, excl] = -float("inf")
            top = logits.topk(10, dim=-1).indices  # [p, 10]
            hits.append((top[:, :, None] == vB[c:c + 512, None, :]).any(-1).float())
        return torch.cat(hits).mean().item()

    for tg in targets:
        print(f"[fbag] sanity {tg}: prec@10 oracle(target) {prec10(vY[tg]):.3f}  "
              f"prior(mean target) {prec10(muy[tg].float()[None].expand(vB.shape[0], -1)):.3f}", flush=True)
    out = {tg: {} for tg in targets}  # best alpha per layer by held-out prec@10 (label-free)
    for l in L:
        Cxx = XtX[l] - n * torch.outer(mu[l], mu[l])
        s, U = torch.linalg.eigh(Cxx)
        s = s.clamp_min(0)
        sbar = s.mean()
        if l == L[0] or l == L[-1]:
            print(f"[fbag] L{l} cov eig: max {s[-1].item():.3g} median {s[d // 2].item():.3g} "
                  f"min {s[0].item():.3g} mean {sbar.item():.3g}", flush=True)
        for tg in targets:
            Cxy = XtY[tg][l] - n * torch.outer(mu[l], muy[tg])
            UtC = U.T @ Cxy
            line, best = [], (-1.0, None, None)
            for a in alphas:
                A = ((U * (1.0 / (s + a * sbar))[None]) @ UtC).float()
                Xc = vX[l] - shift[l] - mu[l].float()
                pred = Xc @ A
                r2 = 1 - ((pred + muy[tg].float() - vY[tg]) ** 2).sum() / ((vY[tg] - muy[tg].float()) ** 2).sum()
                p_full, p_dev = prec10(pred + muy[tg].float()), prec10(pred)
                line.append(f"a={a:g} R2 {r2.item():.3f} p@10 {p_full:.3f} (dev-only {p_dev:.3f})")
                if p_full > best[0]:
                    best = (p_full, a, A)
            out[tg][l] = best[2]
            print(f"[fbag] L{l} {tg}: " + " | ".join(line) + f" -> a={best[1]:g}", flush=True)
            del UtC
    jpp_line = []
    for l in L:
        jpp_line.append(f"L{l} J++ {prec10(unit(vX[l] @ ctx.JPP[l].T) * final_gamma(model)):.3f}")
    print("[fbag] J++ readout prec@10 (same metric): " + " ".join(jpp_line), flush=True)
    g = final_gamma(model)
    print(f"[fbag] gamma min {g.min().item():.3g} max {g.max().item():.3g} mean {g.mean().item():.3g}", flush=True)
    print(f"[fbag] fit done in {time.time() - t_start:.0f}s", flush=True)
    mu = {l: (m.float() + shift[l]) for l, m in mu.items()}
    return out, mu


# ---------------------------------------------------------------- idea 11: far-target Jacobian (random probes)


def fit_jfar(ctx, n_win=4096, T=64, n_tgt=8, kmin=8, batch=8, time_cap=420.0, seed=0, src_lo=16):
    """Random-probe estimate of J_far[l] = E_windows sum_{s} sum_{t in last n_tgt} d h_final[t] / d h_l[s],
    sources s in [src_lo, T - n_tgt - kmin]. One Gaussian u per window placed at every target position;
    g = sum_s (J_s^T u); J_far ~ mean u g^T. Returns {layer: [d_out, d_in]} (full), and two half-sample
    estimates for split-half reliability."""
    from jlens.hooks import ActivationRecorder

    model, dev, L = ctx.model, ctx.device, ctx.readout_layers
    d = model.d_model
    last = len(model.layers) - 1
    win = ctx.wikitext("train", 81920)[-n_win:, :T].long()
    gen = torch.Generator(device=dev).manual_seed(seed)
    M = [{l: torch.zeros(d, d, device=dev) for l in L} for _ in range(2)]
    s_hi = T - n_tgt - kmin  # inclusive
    t_start = time.time()
    n = 0
    for bi, i in enumerate(range(0, n_win, batch)):
        if time.time() - t_start > time_cap:
            break
        ids = win[i:i + batch].to(dev)
        B = ids.shape[0]
        u = torch.randn(B, d, device=dev, generator=gen)
        with ActivationRecorder(model.layers, at=[*L, last], start_graph_at=min(L)) as rec, torch.enable_grad():
            model.forward(ids)
            hT = rec.activations[last]
            cot = torch.zeros_like(hT)
            cot[:, T - n_tgt:] = u[:, None, :].to(cot.dtype)
            grads = torch.autograd.grad(hT, [rec.activations[l] for l in L], cot)
        half = M[bi % 2]
        for l, g in zip(L, grads):
            gs = g[:, src_lo:s_hi + 1].float().sum(1)  # [B, d_in]
            half[l] += u.T @ gs  # [d_out, d_in]
        n += B
        del grads, hT, cot
    full = {l: (M[0][l] + M[1][l]) / n for l in L}
    print(f"[jfar] {n} windows (T={T}, targets last {n_tgt}, sources {src_lo}..{s_hi}) in "
          f"{time.time() - t_start:.0f}s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    return full, M[0], M[1]


def lowrank(Mx, r):
    U, S, Vh = torch.linalg.svd(Mx, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def split_half(ctx, Ma, Mb, R, tag):
    """Label-free reliability: cosine and top-10 overlap (filtered vocab) of the two half estimates on rows R."""
    line = []
    for l in ctx.readout_layers:
        va, vb = R[l] @ Ma[l].T, R[l] @ Mb[l].T
        cos = torch.nn.functional.cosine_similarity(va, vb, dim=-1).mean().item()
        la = unembed_logits(ctx.model, va, ctx.fit.excluded_ids, chunk_size=16)
        lb = unembed_logits(ctx.model, vb, ctx.fit.excluded_ids, chunk_size=16)
        ta, tb = la.topk(10, -1).indices, lb.topk(10, -1).indices
        ov = (ta[:, :, None] == tb[:, None, :]).any(-1).float().mean().item()
        line.append(f"L{l} cos {cos:.2f} ov@10 {ov:.2f}")
    print(f"[{tag}] split-half on fit rows: " + " ".join(line), flush=True)


# ---------------------------------------------------------------- idea 13: interventional lens


class Interv:
    """Write h at block `layer`, position s of K WikiText contexts; read final-block outputs at positions
    t >= s + kmin; return mean over contexts and positions of (patched - mean-patched)."""

    def __init__(self, ctx, K=8, T=48, s=24, kmin=1, n_mu=64, chunk=32, mode="replace", eps=0.1,
                 read_from_s=False, ctx_split="validation"):
        """mode 'add': add eps * rms(resid) * unit(h) at position s (finite-difference JVP: the context-averaged
        Jacobian applied to h), baseline = the unpatched run."""
        self.mode, self.eps, self.r0 = mode, eps, (0 if read_from_s else 1)
        self.model, self.dev = ctx.model, ctx.device
        self.K, self.T, self.s, self.kmin, self.chunk = K, T, s, kmin, chunk
        self.last = len(self.model.layers) - 1
        self.ctx_ids = ctx.wikitext(ctx_split, 512)[-K:, :T].long().to(self.dev)
        # WikiText mean activation per readout layer (positions >= 16 of n_mu windows)
        W = ctx.wikitext("validation", 512)[:n_mu].long().to(self.dev)
        self.mu = {}
        for i in range(0, n_mu, 16):
            rec = record(self.model, W[i:i + 16], ctx.readout_layers)
            for l in ctx.readout_layers:
                self.mu[l] = self.mu.get(l, 0) + rec[l][:, 16:].float().sum((0, 1))
        cnt = n_mu * (W.shape[1] - 16)
        self.mu = {l: v / cnt for l, v in self.mu.items()}
        zero = lambda l: torch.zeros_like(self.mu[l])[None]
        self.base = {l: self.run_pos(self.mu[l][None] if mode == "replace" else zero(l), l)[0]
                     for l in ctx.readout_layers}  # [T-s-1, d]

    @torch.no_grad()
    def run_pos(self, h, layer):
        """h [N, d] -> [N, T-s-1, d]: final-block outputs at positions s+1.., mean over contexts."""
        model, K, s = self.model, self.K, self.s
        outs = []
        for c in range(0, h.shape[0], self.chunk):
            hc = h[c:c + self.chunk]
            n = hc.shape[0]
            ids = self.ctx_ids[None].expand(n, -1, -1).reshape(n * K, -1)
            hh = hc[:, None].expand(-1, K, -1).reshape(n * K, -1)

            def patch(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t = t.clone()
                if self.mode == "replace":
                    t[:, s] = hh.to(t.dtype)
                else:
                    r = t[:, s].float().pow(2).mean(-1, keepdim=True).sqrt()
                    t[:, s] = (t[:, s].float() + self.eps * r * unit(hh.float())).to(t.dtype)
                return t if torch.is_tensor(o) else (t,) + tuple(o[1:])

            hk = model.layers[layer].register_forward_hook(patch)
            try:
                rec = record(model, ids, [self.last])
            finally:
                hk.remove()
            fin = rec[self.last][:, s + self.r0:].float()  # [n*K, T-s-r0, d]
            outs.append(fin.view(n, K, *fin.shape[1:]).mean(1))
        return torch.cat(outs, 0)

    def delta_pos(self, h, layer):
        return self.run_pos(h, layer) - self.base[layer][None]

    def __call__(self, h, layer):
        return self.delta_pos(h, layer)[:, self.kmin - 1:].mean(1)


class IntervLens:
    def __init__(self, JPP, iv, lam=None):
        self.JPP, self.iv, self.lam = JPP, iv, lam

    def transport(self, h, layer, resid_all=None):
        x = self.iv(h, layer)
        if self.lam is None:
            return x
        return unit(h @ self.JPP[layer].T) + self.lam * unit(x)
