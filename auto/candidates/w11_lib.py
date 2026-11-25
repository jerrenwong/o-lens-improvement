"""w11 / Experiment 5A shared code (not a candidate): lenses from per-token probability-gradient directions
(scripts/w11/w11_accum.py tables .hf27/w11/acc_<tag>_<var>.pt).

Token X's score at layer l:  s_X(h) = D_X . h   for covered tokens (N_X >= nmin), otherwise the J-lens direction
g_X = J_l^T (gamma * u_X)  (so s_X = the J-lens logit up to the per-row 1/rms factor, which does not change ranks).
Direction scale (W11_NORM):
   jnorm : D_X / |D_X| * |g_X|     (each token keeps the J-lens direction's norm: only the direction changes)
   unit  : D_X / |D_X|, fallback g_X / |g_X|
   mean  : D_X / W_X (W_X = sum of the context weights: p for p/a/s, shuffled p for d, counts for b/c),
           fallback g_X * c_l, c_l = median over covered tokens of |D_X / W_X| / |g_X|
   fit   : (norm, nmin) chosen jointly on the FIT split
Modes (W11_MODE):
   replace : logits = s(h)
   add     : logits = z(base logits) + beta * z(s(h)),  z = standardise each row over the vocabulary;
             base (W11_BASE) = J (unit(J h)), JPP (unit(J++ h)) or JPPA (J++ + A-lens, w9_alens: ALENS_* env, its lam
             chosen on FIT first). beta chosen on FIT (beta 0 = the base exactly).
   base    : the base alone through this code path (reference; W11_BASE=J/JPP/JPPA)
Selection rule on FIT for every parameter: argmax filtered (pass@1 + pass@10) at the evaluator's --filter level,
ties -> first in grid order.
The harness ranks model.unembed(transport(h)); a per-token lens has no single final-basis vector, so transport returns
h itself (registered by storage pointer with its layer) and model.unembed is wrapped: for a registered tensor it returns
this lens's logits, for anything else the original unembed. Ranking rule, filters, groups and pass@k are the
evaluator's, unchanged.
Env W11_DUMP=<path>: save the raw (unfiltered) top-10 token ids of every scored row per layer (junk check)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import NEG, excl_mask, head_gamma, summarize, unit  # noqa: E402
from olens.readout import candidate_ranks, group_candidate_ranks  # noqa: E402

R = "/workspace/o-lens-improvement"
TAG = {2048: "2b", 4096: "9b", 5120: "27b"}
NMINS = [1, 3, 10, 30, 100]
NORMS = ["jnorm", "unit", "mean"]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]


def zrow(L):
    return (L - L.mean(-1, keepdim=True)) / L.std(-1, keepdim=True).clamp_min(1e-8)


class Table:
    def __init__(self, ctx, prefix, var):
        dev = ctx.device
        t = torch.load(f"{prefix}_{var}.pt", map_location="cpu")
        meta = torch.load(f"{prefix}_meta.pt", map_location="cpu")
        assert torch.equal(t["tok_ids"], meta["tok_ids"])
        self.ids = t["tok_ids"].to(dev)
        self.N = meta["N"].to(dev).float()
        w = {"p": meta["Wp"], "a": meta["Wp"], "s": meta["Wp"], "d": meta["Wd"], "b": meta["N"], "c": meta["N"]}[var]
        self.W = w.to(dev).float()
        self.dir = {l: v.to(dev).float() for l, v in t["dir"].items()}
        self.norm = {l: v.to(dev).float() for l, v in t["norm"].items()}
        self.meta = meta["meta"]


class ProbJ:
    """Per-token lens. cfg: norm, nmin; mode replace/add/base; beta; base callable (h, l) -> final-basis vector."""

    def __init__(self, ctx, table, norm="jnorm", nmin=1, mode="replace", beta=0.0, base=None):
        self.model, self.layers, self.dev = ctx.model, list(ctx.readout_layers), ctx.device
        Wh = ctx.model._lm_head.weight.float()
        gam = head_gamma(ctx.model).float()
        WG = Wh * gam[None]
        self.GJ = {l: WG @ ctx.J[l] for l in self.layers}  # [V, d]: J-lens token directions
        self.gn = {l: self.GJ[l].norm(dim=1) for l in self.layers}
        self.table, self.base, self.mode, self.beta = table, base, mode, beta
        self.reg = {}
        self.set(norm, nmin)

    def set(self, norm, nmin):
        self.norm_kind, self.nmin = norm, nmin
        T = self.table
        keep = (T.N >= nmin)
        keep &= torch.stack([T.norm[l] > 0 for l in self.layers]).all(0)
        self.cov = T.ids[keep]
        self.colscale, self.Dc = {}, {}
        for l in self.layers:
            d = T.dir[l][keep]
            g = self.gn[l][self.cov]
            if norm == "jnorm":
                self.Dc[l] = d * g[:, None]
                self.colscale[l] = None
            elif norm == "z":  # z-score every token's score over reference activations (fallback tokens too)
                self.Dc[l] = d
                self.colscale[l] = "z"
            elif norm == "unit":
                self.Dc[l] = d
                self.colscale[l] = 1.0 / self.gn[l].clamp_min(1e-12)
            else:
                m = d * (T.norm[l][keep] / T.W[keep].clamp_min(1e-12))[:, None]
                self.Dc[l] = m
                self.colscale[l] = (m.norm(dim=1) / g.clamp_min(1e-12)).median()
        if norm == "z":
            self._zstats()
        return self

    def _zstats(self):
        if not hasattr(self, "ref"):
            tag = TAG[self.GJ[self.layers[0]].shape[1]]
            ref = torch.load(os.environ.get("W11_REF") or f"{R}/.hf27/w11/ref_{tag}.pt", map_location="cpu")
            self.ref = {l: ref[l].to(self.dev).float() for l in self.layers}
            self.zJ = {}
            for l in self.layers:
                mu, sd = [], []
                for i in range(0, self.GJ[l].shape[0], 32768):
                    S = self.ref[l] @ self.GJ[l][i:i + 32768].T
                    mu.append(S.mean(0)); sd.append(S.std(0))
                self.zJ[l] = (torch.cat(mu), torch.cat(sd).clamp_min(1e-8))
        self.zc = {}
        for l in self.layers:
            S = self.ref[l] @ self.Dc[l].T
            self.zc[l] = (S.mean(0), S.std(0).clamp_min(1e-8))

    def scores(self, h, l):
        h = h.float()
        L = h @ self.GJ[l].T
        cs = self.colscale[l]
        if isinstance(cs, str):
            L = (L - self.zJ[l][0]) / self.zJ[l][1]
            L[:, self.cov] = (h @ self.Dc[l].T - self.zc[l][0]) / self.zc[l][1]
            return L
        if cs is not None:
            L = L * (cs[None] if cs.dim() else cs)
        L[:, self.cov] = h @ self.Dc[l].T
        return L

    def logits(self, h, l):
        if self.mode == "replace":
            return self.scores(h, l)
        B = orig_unembed(self.model)(unit(self.base(h, l))).float()
        if self.mode == "base" or self.beta == 0:
            return B
        return zrow(B) + self.beta * zrow(self.scores(h, l))

    def transport(self, h, layer, resid_all=None):
        v = h.detach().float().clone()
        self.reg[v.untyped_storage().data_ptr()] = (layer, v)
        return v


# ------------------------------------------------------------------ unembed wrapper

_ORIG = {}


def orig_unembed(model):
    return _ORIG.get(id(model), model.unembed)


def install(model, lens, dump=None):
    if id(model) not in _ORIG:
        if "_unembed_impl" in model.__dict__:
            impl = model.__dict__["_unembed_impl"]
            _ORIG[id(model)] = lambda x: impl(model, x)
        else:
            _ORIG[id(model)] = model.unembed
    orig = _ORIG[id(model)]
    rec = {}

    def find(x):  # registered transport output: same storage, or (after an indexing copy) identical rows
        e = lens.reg.get(x.untyped_storage().data_ptr())
        if e is not None or x.dim() != 2:
            return e
        for layer, v in lens.reg.values():
            if v.shape[1] == x.shape[1] and v.device == x.device and v.dtype == x.dtype:
                if all(bool((v == r).all(-1).any()) for r in x):
                    return (layer, v)
        return None

    def new(x):
        e = find(x)
        if e is None:
            return orig(x)
        L = lens.logits(x, e[0])
        if dump is not None:
            rec.setdefault(e[0], []).append(L.topk(10, dim=-1).indices.cpu())
        return L

    if "_unembed_impl" in model.__dict__:  # lite model: unembed is a method reading this slot
        model.__dict__["_unembed_impl"] = lambda self, x: new(x)
    else:
        model.unembed = new
    return rec


# ------------------------------------------------------------------ FIT scoring


def fit_score(ctx, lens, mask=None):
    mask = excl_mask(ctx) if mask is None else mask
    gid = getattr(ctx.fit, "group_ids", None)
    rk_f, rk_u = {}, {}
    for l in ctx.readout_layers:
        L = lens.logits(ctx.fit.resid[l], l).float()
        rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
        Lm = L.masked_fill(mask.to(L.device), NEG)
        rk_f[l] = (candidate_ranks(Lm, ctx.fit.candidates) if gid is None
                   else group_candidate_ranks(Lm, ctx.fit.candidates, gid))
    return summarize(ctx, rk_f, rk_u)


def per_layer_fit(ctx, lens, mask=None):
    """FIT filtered pass@10 + pass@1 per single layer (to pick a display layer)."""
    from fm27_lib import pass_from_best
    mask = excl_mask(ctx) if mask is None else mask
    gid = getattr(ctx.fit, "group_ids", None)
    out = {}
    for l in ctx.readout_layers:
        L = lens.logits(ctx.fit.resid[l], l).float().masked_fill(mask.to(ctx.device), NEG)
        rk = (candidate_ranks(L, ctx.fit.candidates) if gid is None
              else group_candidate_ranks(L, ctx.fit.candidates, gid))
        out[l] = pass_from_best(ctx, rk, 10)[0] + pass_from_best(ctx, rk, 1)[0]
    return out


def fmt(name, s):
    return f"{name:34s} p10 {s['p10']:.4f} p1 {s['p1']:.4f} | unf p10 {s['p10u']:.4f} p1 {s['p1u']:.4f}"


def choose(res):
    i = max(range(len(res)), key=lambda i: (round(res[i][1]["p1"] + res[i][1]["p10"], 6), -i))
    return res[i][0]


def make_base(ctx, which):
    if which == "J":
        return lambda h, l: h @ ctx.J[l].T.to(h)
    if which == "JPP":
        return lambda h, l: h @ ctx.JPP[l].T.to(h)
    if which == "JPPA":
        import w9_alens
        A = w9_alens.build(ctx)  # lam chosen on FIT (alens rule, evaluator's filter level)
        print(f"[w11] JPPA base: A-lens lam {A.lam}", flush=True)
        return lambda h, l: A.transport(h, l)
    raise ValueError(which)


def build(ctx):
    tag = TAG[ctx.model.d_model]
    mode = os.environ.get("W11_MODE", "replace")
    var = os.environ.get("W11_VAR", "p")
    prefix = os.environ.get("W11_ACC") or f"{R}/.hf27/w11/acc_{tag}"
    norm = os.environ.get("W11_NORM", "fit")
    nmin = os.environ.get("W11_NMIN", "fit")
    bname = os.environ.get("W11_BASE", "J")
    base = make_base(ctx, bname) if mode in ("add", "base") else None
    if mode == "base":
        lens = ProbJ.__new__(ProbJ)
        lens.model, lens.base, lens.mode, lens.beta, lens.reg = ctx.model, base, "base", 0.0, {}
        print(fmt(f"[w11] base {bname}", fit_score(ctx, lens)), flush=True)
    else:
        table = Table(ctx, prefix, var)
        print(f"[w11] table {prefix}_{var}: {len(table.ids)} tokens, meta {table.meta}", flush=True)
        lens = ProbJ(ctx, table, mode="replace", base=base)
        mask = excl_mask(ctx)
        grid = [(n_, m_) for n_ in (NORMS if norm == "fit" else [norm])
                for m_ in (NMINS if nmin == "fit" else [int(nmin)])]
        res = []
        for n_, m_ in grid:
            s = fit_score(ctx, lens.set(n_, m_), mask)
            print(fmt(f"[w11 fit] replace {var} {n_} nmin {m_} cov {len(lens.cov)}", s), flush=True)
            res.append(((n_, m_), s))
        n_, m_ = choose(res)
        lens.set(n_, m_)
        print(f"[w11] chosen norm {n_} nmin {m_} (covered tokens {len(lens.cov)})", flush=True)
        if mode == "add":
            lens.mode = "add"
            fixed = os.environ.get("W11_BETA")
            res = []
            for b_ in ([float(fixed)] if fixed else BETAS):
                lens.beta = b_
                s = fit_score(ctx, lens, mask)
                print(fmt(f"[w11 fit] add {bname} beta {b_}", s), flush=True)
                res.append((b_, s))
            lens.beta = choose(res)
            print(f"[w11] chosen beta {lens.beta}", flush=True)
    dump = os.environ.get("W11_DUMP")
    rec = install(ctx.model, lens, dump)
    if dump:
        import atexit
        atexit.register(lambda: torch.save({l: torch.cat(v) for l, v in rec.items()}, dump))
    return lens
