"""w11b / Experiment 5A-2 shared code (not a candidate): lenses from ALL-p probability-gradient tables
(scripts/w11/w11b_accum.py = M1 probe sketch, scripts/w11/w11b_m2.py = M2 cluster-Jacobian mixture).

Tables cover layers 15/18/21 (2B); other readout layers use the J-lens unchanged.
Token X at a covered layer: score_X(h) = Dhat_X . h, with (W11B_FORM)
   mean   : Dhat = D_X / W_X
   shrink : Dhat = (D_X + kappa * alpha * g_X) / (W_X + kappa),  g_X = J-lens direction J^T (gamma u_X),
            alpha = per-layer median of |D_X / W_X| / |g_X| over the 2000 tokens with largest W (unit match);
            kappa chosen on FIT over KAPPAS (kappa -> inf = the J-lens)
   jnorm  : Dhat = unit(D_X) * |g_X|   (reference)
   add    : z(base logits) + beta * z(scores of the shrink form, kappa chosen on FIT first), W11_BASE = J/JPP/JPPA
Variants (W11B_VAR):
   M1 (W11B_SRC=m1): Dpu (pure sketch), Dhy (hybrid: exact top-10 + sketch tail), Apu / Ahy (weighting only, no
       ubar term), Dsh (shuffled weights, pure sketch), unif (uniform weights, the J-lens through the same sketch:
       B u_X / n_ctx, W = n_ctx)
   M2 (W11B_SRC=m2, W11B_K=8|64): D (mixture), A (weighting only), Dsh (shuffled), pool (cluster sizes as weights =
       the pooled J-lens of the same experts)
Selection on FIT: argmax filtered pass@1 + pass@10 (evaluator filter level), ties -> first in grid order."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w11_lib as W1  # noqa: E402
from fm27_lib import excl_mask, head_gamma  # noqa: E402

R = "/workspace/o-lens-improvement"
KAPPAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def load_tables(ctx, src, var, tag):
    dev = ctx.device
    Wu = ctx.model._lm_head.weight.float()
    gam = head_gamma(ctx.model).float()
    if src == "m1":
        meta = torch.load(f"{R}/.hf27/w11/m1_{tag}_meta.pt", map_location="cpu")
        if var == "unif":
            n = float(meta["n_ctx"])
            D = {l: Wu @ meta["B"][l].to(dev).float().T for l in meta["layers"]}
            W = torch.full((Wu.shape[0],), n, device=dev)
            return D, W, meta["meta"]
        t = torch.load(f"{R}/.hf27/w11/m1_{tag}_{var}.pt", map_location="cpu")
        D = {l: v.to(dev).float() for l, v in t.items()}
        W = (meta["Wsh"] if var == "Dsh" else meta["W"]).to(dev).float()
        return D, W, meta["meta"]
    torch.backends.cuda.matmul.allow_tf32 = True  # 64 [V,d]x[d,d] products per layer (TF32 only while loading)
    try:
        return _load_m2(ctx, var, tag, dev, Wu, gam)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def _load_m2(ctx, var, tag, dev, Wu, gam):
    K = int(os.environ.get("W11B_K", "8"))
    st = torch.load(f"{R}/.hf27/w11/m2_{tag}_K{K}.pt", map_location="cpu")
    ex = torch.load(f"{R}/.hf27/w11/m2fit/experts_K{K}.pt", map_location="cpu", weights_only=False)
    WG = Wu * gam[None]
    D, Wd = {}, None
    for l, w in st["w"].items():
        E = ex["experts"][l].to(dev).float()  # [K, F, N]
        if var == "pool":
            cnt = ex.get("position_counts", {}).get(l)
            cnt = (cnt.to(dev).float() if cnt is not None else torch.ones(K, device=dev))
            D[l] = WG @ (torch.einsum("k,kfn->fn", cnt / cnt.sum(), E))
            Wd = torch.ones(Wu.shape[0], device=dev)
            continue
        ww = (st["wsh"] if var == "Dsh" else st["w"])[l].to(dev).float()  # [V, K]
        acc = torch.zeros(Wu.shape[0], Wu.shape[1], device=dev)
        for k in range(K):
            acc.addcmul_(ww[:, k:k + 1], WG @ E[k])
        if var in ("D", "Dsh"):
            acc -= (st["T2sh"] if var == "Dsh" else st["T2"])[l].to(dev).float()
        D[l] = acc
        Wd = ww.sum(1)
    return D, Wd, st["meta"]


class Lens(W1.ProbJ):
    def __init__(self, ctx, D, W, form="mean", kappa=0.0, mode="replace", beta=0.0, base=None):
        self.model, self.layers, self.dev = ctx.model, list(ctx.readout_layers), ctx.device
        Wh = ctx.model._lm_head.weight.float()
        gam = head_gamma(ctx.model).float()
        WG = Wh * gam[None]
        self.WG, self.J = WG, ctx.J
        self.GJ = {l: WG @ ctx.J[l] for l in D}  # only covered layers (memory); others: (h J^T) WG^T
        self.gn = {l: self.GJ[l].norm(dim=1) for l in D}
        self.Dt, self.Wt = D, W
        self.alpha = {}
        top = W.topk(2000).indices
        for l, Dl in D.items():
            r = (Dl[top].norm(dim=1) / W[top].clamp_min(1e-12)) / self.gn[l][top].clamp_min(1e-12)
            self.alpha[l] = float(r.median())
        self.base, self.mode, self.beta, self.reg = base, mode, beta, {}
        self.set(form, kappa)

    def set(self, form, kappa):
        self.form, self.kappa = form, kappa
        self.G, self.bias = {}, {}
        if form in ("raw", "craw", "cprob", "crel"):
            # literal lenses: D_X (sum over contexts; /n does not change ranks within a row), optionally centred at mu
            n = float(self.n_ctx)
            for l, Dl in self.Dt.items():
                if form == "crel":
                    self.G[l] = Dl / self.Wt.clamp_min(1e-12)[:, None]
                elif form == "cprob":
                    self.G[l] = Dl / n
                    self.bias[l] = self.Wt / n  # pbar_X
                else:
                    self.G[l] = Dl
            return self
        for l, Dl in self.Dt.items():
            if form == "mean":
                G = Dl / self.Wt.clamp_min(1e-12)[:, None]
            elif form == "shrink":
                G = (Dl + kappa * self.alpha[l] * self.GJ[l]) / (self.Wt + kappa).clamp_min(1e-12)[:, None]
            else:  # jnorm
                G = Dl / Dl.norm(dim=1, keepdim=True).clamp_min(1e-30) * self.gn[l][:, None]
            self.G[l] = G
        return self

    def scores(self, h, l):
        h = h.float()
        if l in self.G:
            if self.form in ("craw", "cprob", "crel"):
                h = h - self.mu[l].to(h)
            out = h @ self.G[l].T
            return out + self.bias[l][None] if l in self.bias else out
        return (h @ self.J[l].T) @ self.WG.T


def build(ctx):
    tag = W1.TAG[ctx.model.d_model]
    src = os.environ.get("W11B_SRC", "m1")
    var = os.environ.get("W11B_VAR", "Dhy")
    form = os.environ.get("W11B_FORM", "shrink")
    mode = os.environ.get("W11_MODE", "replace")
    bname = os.environ.get("W11_BASE", "J")
    D, W, meta = load_tables(ctx, src, var, tag)
    n_ctx = meta.get("contexts") if isinstance(meta, dict) else None
    print(f"[w11b] {src} {var} layers {sorted(D)} meta {meta}", flush=True)
    base = W1.make_base(ctx, bname) if mode == "add" else None
    lens = Lens.__new__(Lens)
    lens.n_ctx = n_ctx or 1
    mu = torch.load(f"{R}/.hf27/w11/mu_{tag}.pt", map_location="cpu")["mu"]
    lens.mu = {l: v.to(ctx.device).float() for l, v in mu.items()}
    Lens.__init__(lens, ctx, D, W, base=base)
    print(f"[w11b] alpha {lens.alpha}", flush=True)
    mask = excl_mask(ctx)
    fk = os.environ.get("W11B_KAPPA")
    if form == "shrink" or mode == "add":
        res = []
        for k in ([float(fk)] if fk else KAPPAS):
            s = W1.fit_score(ctx, lens.set("shrink", k), mask)
            print(W1.fmt(f"[w11b fit] shrink kappa {k}", s), flush=True)
            res.append((k, s))
        k = W1.choose(res)
        lens.set("shrink", k)
        print(f"[w11b] chosen kappa {k}", flush=True)
    elif form != "shrink":
        lens.set(form, 0.0)
        print(W1.fmt(f"[w11b fit] {form}", W1.fit_score(ctx, lens, mask)), flush=True)
    if mode == "add":
        lens.mode = "add"
        fb = os.environ.get("W11_BETA")
        res = []
        for b in ([float(fb)] if fb else W1.BETAS):
            lens.beta = b
            s = W1.fit_score(ctx, lens, mask)
            print(W1.fmt(f"[w11b fit] add {bname} beta {b}", s), flush=True)
            res.append((b, s))
        lens.beta = W1.choose(res)
        print(f"[w11b] chosen beta {lens.beta}", flush=True)
    dump = os.environ.get("W11_DUMP")
    rec = W1.install(ctx.model, lens, dump)
    if dump:
        import atexit
        atexit.register(lambda: torch.save({l: torch.cat(v) for l, v in rec.items()}, dump))
    return lens
