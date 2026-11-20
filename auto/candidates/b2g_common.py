"""Shared code for b2g_* candidates (not a candidate itself).

Idea 8: own expert bank with J++-style signed weights.
  Experts per readout layer l (each RMS-normalised):
    jpp     JPP_l h                  j       J_l h                 h      h (logit lens)
    jfq     J_l h minus always-on J-space dictionary features (as jlens_fq)
    jpp_top top-512 SVD part of JPP_l h   jpp_bot  the rest
    jpp_m1  JPP_{l-1} resid[l-1]     jpp_p1  JPP_{l+1} resid[l+1]
  Lens: v = sum_e w_le unit(expert_e). w_jpp fixed at 1 (rank is scale-free), the other weights signed,
  ridge lam * ||w_rest||^2 toward plain J++. Objective: J++'s surrogate (sigmoid of best-candidate logit minus
  10th largest logit, in units of temperature x logit std), per layer, per-eval mean then mean over evals,
  on ctx.fit only. Final norm is RMSNorm, so logits of the weighted sum are a positive rescaling of the
  weighted sum of the experts' logits: logits are cached once per (layer, item, expert) on a per-item
  vocabulary subset (union of every expert's top-256, filtered, plus the candidates).
  lam is chosen by 5-fold CV (stratified by task, 3 repeats) on the fit split, on the benchmark metric
  (best rank over layers, item-weighted macro pass@10). Then refit on all 75.

Idea 5: layer-increment lens unit(P_l) - a unit(P_{l-k}) (fit sweep, exact ranks).
"""

import copy
import time
from collections import defaultdict

import torch

from olens.bsf_io import BlockDict
from olens.readout import candidate_ranks, unembed_logits

BANK8 = ["jpp", "j", "h", "jfq", "jpp_top", "jpp_bot", "jpp_m1", "jpp_p1"]
BANK3 = ["jpp", "jfq", "h"]
TOP_R = 512
TAU = 0.1
LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def log(*a):
    print("[b2g]", *a, flush=True)


# ---------------------------------------------------------------- exact fit scoring (full vocabulary)


def fit_pass(ctx, vecs, k=10):
    """vecs {layer: [n_fit, d]} -> (macro pass@k, per task), benchmark rule."""
    best = [dict() for _ in ctx.fit.candidates]
    for layer, v in vecs.items():
        logits = unembed_logits(ctx.model, v, ctx.fit.excluded_ids, chunk_size=16)
        for p, d in enumerate(candidate_ranks(logits, ctx.fit.candidates)):
            for kk, r in d.items():
                best[p][kk] = min(best[p].get(kk, 10 ** 9), r)
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(v) / len(v) for t, v in sorted(per.items())}
    return sum(task.values()) / len(task), task


def report(ctx, name, lens):
    vecs = {l: lens.transport(ctx.fit.resid[l], l, ctx.fit.resid) for l in ctx.readout_layers}
    p10, t10 = fit_pass(ctx, vecs, 10)
    p1, _ = fit_pass(ctx, vecs, 1)
    log(f"[fit exact] {name:28s} p10 {p10:.4f} p1 {p1:.4f} | " + " ".join(f"{t[:5]} {v:.2f}" for t, v in t10.items()))
    return p10, p1


# ---------------------------------------------------------------- expert bank


class Bank:
    def __init__(self, ctx, names=BANK8):
        self.names, self.J, self.JPP = list(names), ctx.J, ctx.JPP
        self.dicts, self.keep, self.svd = {}, {}, {}
        if "jfq" in names:
            for layer in ctx.readout_layers:
                path = ctx.dict_path(layer, "jspace")
                self.dicts[layer] = BlockDict.load(path, ctx.device)
                st = torch.load(path[:-3] + "_stats.pt", map_location=ctx.device, weights_only=True)
                self.keep[layer] = st["fire_count"].float() / float(st["rows"]) < TAU
        if "jpp_top" in names or "jpp_bot" in names:
            for layer in ctx.readout_layers:
                U, S, Vh = torch.linalg.svd(self.JPP[layer].float())
                self.svd[layer] = ((U[:, :TOP_R] * S[:TOP_R]).contiguous(), Vh[:TOP_R].T.contiguous())

    def expert(self, name, h, layer, R):
        if name == "jpp":
            return h @ self.JPP[layer].T
        if name == "j":
            return h @ self.J[layer].T
        if name == "h":
            return h
        if name == "jfq":
            x = h @ self.J[layer].T
            D, keep = self.dicts[layer], self.keep[layer]
            z = D.encode(x)
            return x - (D.decode(z) - D.decode(z * keep[None, :, None]))
        if name in ("jpp_top", "jpp_bot"):
            US, V = self.svd[layer]
            top = (h @ V) @ US.T
            return top if name == "jpp_top" else h @ self.JPP[layer].T - top
        if name == "jpp_m1":
            return R[layer - 1] @ self.JPP[layer - 1].T
        if name == "jpp_p1":
            return R[layer + 1] @ self.JPP[layer + 1].T
        raise KeyError(name)

    def outputs(self, h, layer, R):
        return torch.stack([unit(self.expert(n, h, layer, R)) for n in self.names], 1)  # [N, E, d]


class WeightedLens:
    def __init__(self, bank, W):
        self.bank, self.W = bank, W  # W {layer: [E]} over bank.names

    def transport(self, h, layer, resid_all=None):
        X = self.bank.outputs(h, layer, resid_all)
        return torch.einsum("ned,e->nd", X, self.W[layer].to(X))


# ---------------------------------------------------------------- logit cache on vocabulary subsets


class Cache:
    def __init__(self, ctx, bank, T=256):
        fit, model, dev = ctx.fit, ctx.model, ctx.device
        self.layers = list(ctx.readout_layers)
        P = len(fit.candidates)
        self.P = P
        q_item, q_cands = [], []
        for p, cand in enumerate(fit.candidates):
            for _, ids in cand.items():
                q_item.append(p)
                q_cands.append(list(ids))
        self.q_item = torch.tensor(q_item, device=dev)
        tasks = sorted(set(fit.eval_slugs))
        self.tasks = tasks
        self.item_task = torch.tensor([tasks.index(s) for s in fit.eval_slugs], device=dev)
        self.Q = len(q_item)
        excl = None
        subs, Ls, Gs, ms = [], [], [], []
        for layer in self.layers:
            X = bank.outputs(fit.resid[layer], layer, fit.resid)  # [P, E, d]
            E = X.shape[1]
            L = torch.cat([model.unembed(X[i:i + 16].reshape(-1, X.shape[-1])).float()
                           for i in range(0, P, 16)], 0).view(P, E, -1)
            V = L.shape[-1]
            if excl is None:
                excl = torch.zeros(V, dtype=torch.bool, device=L.device)
                excl[ctx.fit.excluded_ids.to(L.device).long()] = True
            ms.append(L.mean(-1))
            Gs.append(torch.einsum("pev,pfv->pef", L, L) / V)
            top = L.masked_fill(excl, float("-inf")).topk(T, dim=-1).indices  # [P, E, T]
            per = []
            for p in range(P):
                extra = torch.tensor([i for q, c in zip(q_item, q_cands) if q == p for i in c],
                                     device=L.device, dtype=torch.long)
                per.append(torch.unique(torch.cat([top[p].flatten(), extra])))
            subs.append(per)
            Ls.append((L, per))
            del X
        S = max(len(s) for per in subs for s in per)
        nl = len(self.layers)
        E = len(bank.names)
        self.E = E
        self.sub = torch.zeros(nl, P, S, dtype=torch.long, device=dev)
        self.valid = torch.zeros(nl, P, S, dtype=torch.bool, device=dev)
        self.Lsub = torch.zeros(nl, P, E, S, device=dev)
        for li, (L, per) in enumerate(Ls):
            for p, s in enumerate(per):
                n = len(s)
                self.sub[li, p, :n] = s
                self.valid[li, p, :n] = True
                self.Lsub[li, p, :, :n] = L[p][:, s]
        del Ls
        self.G = torch.stack(Gs)  # [nl, P, E, E]
        self.m = torch.stack(ms)  # [nl, P, E]
        # candidate mask per pair
        self.C = torch.zeros(nl, self.Q, S, dtype=torch.bool, device=dev)
        for q, (p, c) in enumerate(zip(q_item, q_cands)):
            ct = torch.tensor(c, device=dev)
            self.C[:, q] = torch.isin(self.sub[:, p], ct) & self.valid[:, p]
        assert bool(self.C.any(-1).all()), "candidate missing from subset"
        self.S = S
        log(f"cache: {nl} layers, P {P}, Q {self.Q}, E {E}, S {S}")

    # w: [B, nl, E] -> logits [B, nl, P, S]
    def logits(self, w):
        x = torch.einsum("lpes,ble->blps", self.Lsub, w)
        return x.masked_fill(~self.valid, float("-inf"))

    def spread(self, w):
        var = torch.einsum("ble,lpef,blf->blp", w, self.G, w) - torch.einsum("ble,lpe->blp", w, self.m) ** 2
        return var.clamp_min(1e-12).sqrt()

    def ranks(self, w):
        """[B, nl, Q] subset ranks (1 + #greater)."""
        x = self.logits(w)[:, :, self.q_item]  # [B, nl, Q, S]
        best = x.masked_fill(~self.C, float("-inf")).amax(-1)
        return 1 + (x > best[..., None]).sum(-1)

    def macro(self, rq, k, items=None):
        """rq [B, Q] best rank over layers -> [B] item-weighted macro pass@k (over `items` mask [P])."""
        hit = (rq <= k).float()
        B = hit.shape[0]
        s = torch.zeros(B, self.P, device=hit.device).index_add_(1, self.q_item, hit)
        n = torch.zeros(self.P, device=hit.device).index_add_(0, self.q_item, torch.ones(self.Q, device=hit.device))
        share = s / n
        if items is None:
            items = torch.ones(self.P, dtype=torch.bool, device=hit.device)
        out = []
        for t in range(len(self.tasks)):
            m = items & (self.item_task == t)
            if m.any():
                out.append(share[:, m].mean(1))
        return torch.stack(out, 1).mean(1)

    def soft(self, w, train_items, temp=0.1, k=10):
        """J++ surrogate per layer: [B, nl]; pairs restricted to train items; per-eval mean then mean."""
        x = self.logits(w)
        sp = self.spread(w).detach()
        xq = x[:, :, self.q_item]
        best = xq.masked_fill(~self.C, float("-inf")).amax(-1)
        kth = xq.topk(k, dim=-1).values[..., -1]
        s = torch.sigmoid((best - kth) / (temp * sp[:, :, self.q_item]))  # [B, nl, Q]
        qm = train_items[self.q_item]
        qt = self.item_task[self.q_item]
        out = []
        for t in range(len(self.tasks)):
            m = qm & (qt == t)
            if m.any():
                out.append(s[..., m].mean(-1))
        return torch.stack(out, -1).mean(-1)


def full_w(cache, eidx, delta, shared):
    """delta [B, Lw, len(eidx)-1] -> w [B, nl, E] (jpp = eidx[0] fixed at 1)."""
    B = delta.shape[0]
    nl = len(cache.layers)
    d = delta.expand(B, nl, -1) if shared else delta
    w = torch.zeros(B, nl, cache.E, device=delta.device)
    w[:, :, eidx[0]] = 1.0
    w = w.index_copy(2, torch.tensor(eidx[1:], device=delta.device), d) if len(eidx) > 1 else w
    return w


def fit_weights(cache, eidx, shared, lams, train_items, steps=200, lr=0.05):
    dev = cache.Lsub.device
    B = len(lams)
    lam = torch.tensor(lams, device=dev)
    Lw = 1 if shared else len(cache.layers)
    delta = torch.zeros(B, Lw, len(eidx) - 1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        with torch.enable_grad():
            w = full_w(cache, eidx, delta, shared)
            sl = cache.soft(w, train_items)  # [B, nl]
            pen = delta.pow(2).sum(-1)  # [B, Lw]
            if shared:
                obj = sl.mean(1) - lam * pen[:, 0]
            else:
                obj = (sl - lam[:, None] * pen).sum(1)
            opt.zero_grad()
            (-obj.sum()).backward()
        opt.step()
    return full_w(cache, eidx, delta.detach(), shared)


def folds(cache, n_folds=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(cache.P, dtype=torch.long)
    it = cache.item_task.cpu()
    for t in range(len(cache.tasks)):
        idx = (it == t).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        f[idx] = torch.arange(len(idx)) % n_folds
    return f.to(cache.Lsub.device)


def cv(cache, eidx, shared, lams=LAMS, reps=3, n_folds=5):
    """-> (cv p10 [B], cv p1 [B], jpp p10, jpp p1), mean over repeats of pooled held-out predictions."""
    dev = cache.Lsub.device
    B = len(lams)
    p10s, p1s = [], []
    for rep in range(reps):
        f = folds(cache, n_folds, seed=rep)
        R = torch.zeros(B, cache.Q, dtype=torch.long, device=dev)
        for k in range(n_folds):
            tr = f != k
            w = fit_weights(cache, eidx, shared, lams, tr)
            with torch.no_grad():
                r = cache.ranks(w).amin(1)  # [B, Q]
            te = (~tr)[cache.q_item]
            R[:, te] = r[:, te]
        p10s.append(cache.macro(R, 10))
        p1s.append(cache.macro(R, 1))
    w0 = full_w(cache, eidx, torch.zeros(1, 1, len(eidx) - 1, device=dev), True)
    r0 = cache.ranks(w0).amin(1)
    return (torch.stack(p10s).mean(0), torch.stack(p1s).mean(0),
            float(cache.macro(r0, 10)), float(cache.macro(r0, 1)))


def bank_lens(ctx, bank_names, shared, cv_table=True):
    """Build the bank, CV lam for (bank_names, shared), refit on all fit items, return the lens."""
    t0 = time.time()
    bank = Bank(ctx, BANK8)
    cache = Cache(ctx, bank)
    log(f"bank+cache built in {time.time() - t0:.0f}s")
    # stand-alone expert scores (subset ranks, sanity)
    eye = torch.eye(cache.E, device=ctx.device)[:, None, :].expand(-1, len(cache.layers), -1)
    sa = cache.macro(cache.ranks(eye).amin(1), 10)
    log("stand-alone fit p10: " + " ".join(f"{n} {v:.3f}" for n, v in zip(BANK8, sa.tolist())))
    variants = {(tuple(bank_names), shared)}
    if cv_table:
        variants |= {(tuple(b), s) for b in (BANK8, BANK3) for s in (True, False)}
    chosen = None
    for bn, sh in sorted(variants):
        eidx = [BANK8.index(n) for n in bn]
        t1 = time.time()
        c10, c1, j10, j1 = cv(cache, eidx, sh)
        tag = f"bank{len(bn)} {'shared' if sh else 'perlayer'}"
        log(f"CV {tag}: plain J++ p10 {j10:.4f} p1 {j1:.4f} | " +
            " ".join(f"lam {l:g}: {a:.4f}/{b:.4f}" for l, a, b in zip(LAMS, c10.tolist(), c1.tolist())) +
            f" ({time.time() - t1:.0f}s)")
        if bn == tuple(bank_names) and sh == shared:
            score = [(a, b, l) for a, b, l in zip(c10.tolist(), c1.tolist(), LAMS)]
            best = max(score)
            chosen = (eidx, best)
            log(f"chosen {tag}: lam {best[2]:g} cv p10 {best[0]:.4f} p1 {best[1]:.4f} vs plain J++ "
                f"{j10:.4f}/{j1:.4f} -> {'BEATS' if best[0] > j10 else 'does NOT beat'} J++ on CV")
    eidx, (_, _, lam) = chosen
    all_items = torch.ones(cache.P, dtype=torch.bool, device=ctx.device)
    w = fit_weights(cache, eidx, shared, [lam], all_items)[0]  # [nl, E]
    r = cache.ranks(w[None]).amin(1)
    log(f"refit all 75 (subset ranks): p10 {float(cache.macro(r, 10)):.4f} p1 {float(cache.macro(r, 1)):.4f}")
    for li, layer in enumerate(cache.layers):
        log(f"w L{layer}: " + " ".join(f"{BANK8[e]} {float(w[li, e]):+.3f}" for e in eidx))
    names = [BANK8[e] for e in eidx]
    sub_bank = copy.copy(bank)
    sub_bank.names = names
    W = {layer: w[li, eidx].clone() for li, layer in enumerate(cache.layers)}
    lens = WeightedLens(sub_bank, W)
    del cache
    torch.cuda.empty_cache()
    jb = copy.copy(bank)
    jb.names = ["jpp"]
    report(ctx, "plain J++", WeightedLens(jb, {l: torch.ones(1, device=ctx.device) for l in ctx.readout_layers}))
    report(ctx, "fitted bank", lens)
    log(f"build total {time.time() - t0:.0f}s")
    return lens


# ---------------------------------------------------------------- idea 5: layer increment


class IncLens:
    def __init__(self, JPP, a, k):
        self.JPP, self.a, self.k = JPP, a, k

    def transport(self, h, layer, resid_all=None):
        lo = layer - self.k
        return unit(h @ self.JPP[layer].T) - self.a * unit(resid_all[lo] @ self.JPP[lo].T)
