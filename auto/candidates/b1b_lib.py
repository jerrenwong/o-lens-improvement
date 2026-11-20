"""Shared code for b1b_* candidates (J++ minus input-independent / token-type priors).

Not a candidate itself. Each b1b_<variant>.py calls `build_variant(ctx, kind, **kw)`.
kinds:
  mean      v = unit(JPP_l h) - alpha * m_l        (m_l = WikiText mean of unit(JPP_l h), all positions >= 1)
  bnd       same, mean over sentence-final period / newline positions only
  prior     v = unit(JPP_l h) + b * f              (f: ridge fit of a per-token score onto unembedding rows,
                                                    b chosen on the FIT split from a small grid)
  sametok   v = unit(JPP_l h) - alpha * m_{tok,l}  (tok identified from resid_all[0] by nearest WikiText
                                                    layer-0 centroid; unknown tokens: alpha = 0)
"""

import torch

from olens.bsf_train import harvest

N_WIN, BATCH, HOLDOUT = 2048, 64, 64
TRACK_MIN, TRACK_MAX = 30, 6000


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


# ---------------------------------------------------------------- WikiText statistics


def collect(ctx, need_tok=False):
    model, JPP, layers = ctx.model, ctx.JPP, ctx.readout_layers
    dev = ctx.device
    W = ctx.wikitext("train", N_WIN).long()
    tr, ho = W[:-HOLDOUT], W[-HOLDOUT:]
    ids_tr = tr[:, 1:]
    V = model._lm_head.weight.shape[0]
    counts = torch.bincount(ids_tr.flatten(), minlength=V).float()
    tok = model.tokenizer

    # boundary ids: tokens ending a sentence with a period, or containing a newline
    present = torch.nonzero(counts > 0).flatten().tolist()
    bnd = torch.zeros(V, dtype=torch.bool)
    for i in present:
        s = tok.decode([i])
        if "\n" in s or s.rstrip().endswith("."):
            bnd[i] = True
    bnd = bnd.to(dev)

    tracked = torch.nonzero(counts >= TRACK_MIN).flatten()
    tracked = tracked[counts[tracked].argsort(descending=True)][:TRACK_MAX]
    K = len(tracked)
    slot_of = torch.full((V,), -1, dtype=torch.long)
    slot_of[tracked] = torch.arange(K)
    slot_of = slot_of.to(dev)

    d = JPP[layers[0]].shape[0]
    g_sum = {l: torch.zeros(d, device=dev) for l in layers}
    b_sum = {l: torch.zeros(d, device=dev) for l in layers}
    t_sum = {l: torch.zeros(K, d, device=dev) for l in layers} if need_tok else None
    h0_sum = torch.zeros(K, d, device=dev) if need_tok else None
    t_cnt = torch.zeros(K, device=dev)
    n_all, n_bnd = 0, 0
    hl = [0] + list(layers)
    for i in range(0, tr.shape[0], BATCH):
        ids = tr[i:i + BATCH].to(dev)
        acts = harvest(model, ids, hl)
        flat = ids[:, 1:].reshape(-1)
        isb = bnd[flat]
        slot = slot_of[flat]
        m = slot >= 0
        n_all += flat.numel()
        n_bnd += int(isb.sum())
        t_cnt.index_add_(0, slot[m], torch.ones(int(m.sum()), device=dev))
        if need_tok:
            h0_sum.index_add_(0, slot[m], unit(acts[0].float())[m])
        for l in layers:
            u = unit(acts[l].float() @ JPP[l].T)
            g_sum[l] += u.sum(0)
            b_sum[l] += u[isb].sum(0)
            if need_tok:
                t_sum[l].index_add_(0, slot[m], u[m])
        del acts
    out = dict(
        V=V, counts=counts, bnd=bnd, n_all=n_all, n_bnd=n_bnd,
        mean={l: g_sum[l] / n_all for l in layers},
        mean_bnd={l: b_sum[l] / max(n_bnd, 1) for l in layers},
    )
    print(f"[b1b] wikitext positions {n_all}, boundary positions {n_bnd}, tracked tokens {K}")
    if need_tok:
        c = t_cnt.clamp_min(1)[:, None]
        out["tok_mean"] = {l: t_sum[l] / c for l in layers}
        cent = h0_sum / c
        out["cent"] = cent / cent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out["tracked"] = tracked.to(dev)
        # token-identification check on held-out windows
        acts = harvest(model, ho.to(dev), [0])
        flat = ho[:, 1:].reshape(-1).to(dev)
        sl, cs = identify(out["cent"], acts[0].float())
        truth = slot_of[flat]
        for tau in (0.5, 0.7, 0.8, 0.9):
            ok = cs >= tau
            known = truth >= 0
            acc = ((sl == truth) & ok & known).sum() / known.sum()
            wrong = (ok & (sl != truth)).sum() / ok.sum().clamp_min(1)
            print(f"[b1b] token-id holdout tau={tau}: recall(tracked)={acc:.3f} "
                  f"wrong-among-assigned={wrong:.3f} assigned={ok.float().mean():.3f}")
    return out


def identify(cent, h0):
    x = h0 / h0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cs, sl = (x @ cent.T).max(dim=-1)
    return sl, cs


# ---------------------------------------------------------------- logit-space prior


def logit_direction(ctx, score):
    """f (final basis, rms 1) with unembed logits of (g * f) ~ score (ridge over all vocab rows)."""
    model = ctx.model
    Wh = model._lm_head.weight.float()  # [V, d]
    d = Wh.shape[1]
    g = model._final_norm(torch.ones(1, d, device=Wh.device, dtype=model._lm_head.weight.dtype)).float()[0]
    s = score.to(Wh.device).float()
    s = s - s.mean()
    G = Wh.T @ Wh
    lam = 1e-3 * G.diagonal().mean()
    fp = torch.linalg.solve(G + lam * torch.eye(d, device=G.device), Wh.T @ s)
    pred = Wh @ fp
    r2 = 1 - ((pred - s).pow(2).sum() / s.pow(2).sum())
    print(f"[b1b] prior ridge R^2 = {float(r2):.3f}")
    f = fp / g
    return unit(f).to(ctx.device)


def word_score(ctx, V):
    tok = ctx.model.tokenizer
    s = torch.zeros(V)
    for i in range(V):
        t = tok.decode([i])
        w = t[1:]
        if t.startswith(" ") and len(w) >= 3 and w.isascii() and w.isalpha():
            s[i] = 1.0
    print(f"[b1b] whole-word tokens: {int(s.sum())} of {V}")
    return s


# ---------------------------------------------------------------- fit scoring and diagnostics


def fit_score(ctx, lens):
    fit, model = ctx.fit, ctx.model
    excl = fit.excluded_ids.to(ctx.device)
    best = {}
    for l in ctx.readout_layers:
        v = lens.transport(fit.resid[l].clone(), l, {k: x.clone() for k, x in fit.resid.items()})
        lg = model.unembed(v).float()
        lg[:, excl.to(lg.device)] = float("-inf")
        for i, cands in enumerate(fit.candidates):
            for name, ids in cands.items():
                bl = lg[i, torch.as_tensor(ids, device=lg.device)].max()
                r = int((lg[i] > bl).sum()) + 1
                best[(i, name)] = min(best.get((i, name), 10**9), r)
    out = {}
    for k in (1, 10):
        per = {}
        for (i, name), r in best.items():
            per.setdefault(fit.eval_slugs[i], []).append(float(r <= k))
        per = {e: sum(v) / len(v) for e, v in per.items()}
        out[k] = (sum(per.values()) / len(per), per)
    return out


def top_tokens(ctx, v, k=10, excl=True):
    lg = ctx.model.unembed(v[None]).float()[0]
    if excl:
        lg[ctx.fit.excluded_ids.to(lg.device)] = float("-inf")
    return [ctx.model.tokenizer.decode([int(i)]) for i in lg.topk(k).indices]


# ---------------------------------------------------------------- lenses


class Lens:
    def __init__(self, ctx, kind, alpha=0.0, vec=None, tok=None, tau=0.8):
        self.JPP, self.kind, self.alpha, self.vec, self.tok, self.tau = ctx.JPP, kind, alpha, vec, tok, tau

    def transport(self, h, layer, resid_all=None):
        u = unit(h @ self.JPP[layer].T)
        if self.kind in ("mean", "bnd"):
            return u - self.alpha * self.vec[layer][None]
        if self.kind == "prior":
            return u + self.alpha * self.vec[None]
        if self.kind == "sametok":
            sl, cs = identify(self.tok["cent"], resid_all[0].float())
            m = self.tok["tok_mean"][layer][sl]
            a = self.alpha * (cs >= self.tau).float()[:, None]
            return u - a * m
        return u


def report(ctx, lens, tag):
    sc = fit_score(ctx, lens)
    print(f"[b1b] FIT {tag}: pass10 {sc[10][0]:.3f} pass1 {sc[1][0]:.3f} per-task "
          + " ".join(f"{e[:5]}={v:.2f}" for e, v in sorted(sc[10][1].items())))
    return sc


def build_variant(ctx, kind, alpha=1.0, score=None, grid=None, tau=0.8):
    torch.manual_seed(0)
    base = Lens(ctx, "none")
    report(ctx, base, "J++ (unit)")
    if kind in ("mean", "bnd"):
        st = collect(ctx)
        vec = st["mean"] if kind == "mean" else st["mean_bnd"]
        for l in ctx.readout_layers:
            print(f"[b1b] L{l} |m|_rms={float(vec[l].pow(2).mean().sqrt()):.3f} top10(m):",
                  top_tokens(ctx, vec[l]))
        lens = Lens(ctx, kind, alpha, vec=vec)
        # junk check on fit association rows at L28 / L20
        idx = [i for i, e in enumerate(ctx.fit.eval_slugs) if e.startswith("assoc")][:3]
        for l in (20, 28):
            for i in idx:
                h = ctx.fit.resid[l][i:i + 1]
                print(f"[b1b] fit assoc {i} L{l} before:", top_tokens(ctx, base.transport(h, l)[0]))
                print(f"[b1b] fit assoc {i} L{l} after :", top_tokens(ctx, lens.transport(h, l)[0]))
        report(ctx, lens, f"{kind} alpha={alpha}")
        return lens
    if kind == "prior":
        V = ctx.model._lm_head.weight.shape[0]
        if score == "logfreq":
            st = collect(ctx)
            s = torch.log1p(st["counts"])
        else:
            s = word_score(ctx, V)
        f = logit_direction(ctx, s)
        print("[b1b] top10(+f):", top_tokens(ctx, f), " top10(-f):", top_tokens(ctx, -f))
        best = None
        for b in grid:
            lens = Lens(ctx, "prior", b, vec=f)
            sc = report(ctx, lens, f"prior {score} b={b}")
            key = (round(sc[10][0], 4), round(sc[1][0], 4), -abs(b))
            if best is None or key > best[0]:
                best = (key, b)
        print(f"[b1b] chosen b = {best[1]}")
        return Lens(ctx, "prior", best[1], vec=f)
    if kind == "sametok":
        st = collect(ctx, need_tok=True)
        lens = Lens(ctx, "sametok", alpha, tok=st, tau=tau)
        sl, cs = identify(st["cent"], ctx.fit.resid[0].float())
        ids = st["tracked"][sl]
        names = [repr(ctx.model.tokenizer.decode([int(i)])) if c >= tau else "?" for i, c in zip(ids, cs)]
        print("[b1b] fit readout tokens:", list(zip([e[:5] for e in ctx.fit.eval_slugs], names)))
        report(ctx, lens, f"sametok alpha={alpha} tau={tau}")
        return lens
    raise ValueError(kind)
