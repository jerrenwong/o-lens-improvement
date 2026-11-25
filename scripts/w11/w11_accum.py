"""w11 / Experiment 5A: per-token PROBABILITY-gradient directions (pod).

Position convention = the J-lens fit's (workspace_lens.fitting.jacobian_fitting, olens.train.model_pass):
valid positions 16..S-2 of a 128-token window; a source position t at layer l is paired with EVERY valid target
position t' >= t (the cotangent is placed at all valid targets at once, the source gradient is read at every valid
source and summed).  Here a "context" c = (window, valid target t').  For every context, the top-K non-junk tokens X
of the model's own distribution p(t') (true probabilities, not renormalised) get

  later-target ("lt") variants, gradient summed over valid sources t <= t' (one backward row per (window, X, variant);
  the row's scalar is  sum over the window's contexts where X is in the top-K of  w_c * f_X(c)):
     p : f = p_X,              w = 1             D_X  = sum_c p_X(c) J_c^T (u_X - ubar_c)       (main object)
     a : f = logit_X,          w = p_X(c) (const) D'_X = sum_c p_X(c) J_c^T u_X                  (weighting only)
     b : f = log p_X,          w = 1             sum_c J_c^T (u_X - ubar_c)                      (subtraction only)
     c : f = logit_X,          w = 1             sum_c J_c^T u_X                                 (uniform control)
     d : f = log p_X,          w = shuffled p    weights drawn (with replacement) from the token's own p values over
                                                 all contexts of the run (pre-pass): shuffled-weight control
  same-position ("s") variant of p: one row per (context, X), gradient read at source t = t' only.
J_c^T includes the final RMSNorm (autograd through the real norm; cotangent built at the norm output as
c(t') = a(t') u_X - b(t') ubar(t'), ubar = p^T W, then pulled through the norm with autograd).

    python scripts/w11/w11_accum.py --model Qwen/Qwen3.5-2B --budget-s 7200 --out .hf27/w11/acc_2b
    python scripts/w11/w11_accum.py --model Qwen/Qwen3.5-2B --sanity     (identity / composition checks only)
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(R / "auto"))
import filters as FLT  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

LT = ["p", "a", "b", "c", "d"]
SAME = [True]
SKIP = 16
LAYERS = {"Qwen/Qwen3.5-2B": [3, 6, 9, 12, 15, 18, 21], "Qwen/Qwen3.5-9B": [4, 8, 12, 16, 20, 24, 28]}


def junk_mask(tok, V):
    texts = FLT.decode_all(tok, V)
    spec = FLT.special_ids(tok)
    m = torch.zeros(V, dtype=torch.bool)
    for i, t in enumerate(texts):
        if (not any(ch.isalnum() for ch in t)) or i in spec or any(ch in t for ch in "@&<>"):
            m[i] = True
    return m


def make_windows(tok, S, n_max, seed=0):
    """Alternate (1) the J-lens/J++ fit pool (runs/r1_2b/prompts_raw.json, WikiText-103 train paragraphs, first S
    tokens) and (2) runs/w9/prefixes_100k.json passages (rows >= 2000, shuffled) packed with blank lines into S-token
    windows."""
    jp = json.load(open(R / "runs/r1_2b/prompts_raw.json"))
    A = []
    for p in jp:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) >= S:
            A.append(ids[:S])
    pre = json.load(open(R / "runs/w9/prefixes_100k.json"))[2000:]
    rnd = random.Random(seed)
    order = list(range(len(pre)))
    rnd.shuffle(order)
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    B, cur, srcs, curs = [], [], [], set()
    for i in order:
        ids = tok(pre[i]["text"], add_special_tokens=False)["input_ids"]
        cur = cur + (sep if cur else []) + ids
        curs.add(pre[i]["src"])
        if len(cur) >= S:
            B.append(cur[:S]); srcs.append(sorted(curs)); cur, curs = [], set()
        if len(B) >= n_max:
            break
    out, kinds = [], []
    for i in range(max(len(A), len(B))):
        for lst, k in ((A, "jfit"), (B, "prefix")):
            if i < len(lst):
                out.append(lst[i]); kinds.append(k)
    return out[:n_max], kinds[:n_max]


class Acc:
    def __init__(self, model, layers, K, B, cap, dev, junk, pool=None):
        self.m, self.layers, self.K, self.B, self.dev = model, layers, K, B, dev
        self.tgt = model.n_layers - 1
        self.W = model._lm_head.weight.detach().float()  # [V, d]
        self.norm = copy.deepcopy(model._final_norm).float()
        for p_ in self.norm.parameters():
            p_.requires_grad_(False)
        self.junk = junk.to(dev)
        d = self.W.shape[1]
        self.cap = cap
        self.row = torch.full((self.W.shape[0],), -1, dtype=torch.long, device=dev)
        self.ids = []
        self.same = True
        self.D = {v: {l: torch.zeros(cap, d, device=dev) for l in layers} for v in LT + (["s"] if SAME[0] else [])}
        self.N = torch.zeros(cap, device=dev, dtype=torch.float64)
        self.Wp = torch.zeros(cap, device=dev, dtype=torch.float64)
        self.Wp2 = torch.zeros(cap, device=dev, dtype=torch.float64)
        self.Wd = torch.zeros(cap, device=dev, dtype=torch.float64)
        self.pool = pool  # (vals [n], start [V], len [V]) on dev, or None
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(0)

    # ---------------------------------------------------------------- model pieces
    def forward(self, ids):
        S = ids.shape[0]
        with ActivationRecorder(self.m.layers, at=[*self.layers, self.tgt], start_graph_at=min(self.layers)) as rec, \
                torch.enable_grad():
            self.m.forward(ids[None].expand(self.B, -1))
            hT = rec.activations[self.tgt]
            srcs = [rec.activations[l] for l in self.layers]
        return hT, srcs

    def dist(self, hF):  # hF [S, d] (any dtype) -> p [S, V] fp32
        with torch.no_grad():
            z = self.norm(hF.float())
            return torch.softmax(z @ self.W.T, dim=-1)

    def topk(self, p, valid):
        pv = p[valid].masked_fill(self.junk[None], -1.0)
        val, tok = pv.topk(self.K, dim=-1)
        return tok, val  # [T, K]

    def norm_vjp(self, hF, c):  # c [b, S, d] cotangent on the norm output -> cotangent on hF
        with torch.enable_grad():
            h = hF.float()[None].expand(c.shape[0], -1, -1).clone().requires_grad_(True)
            z = self.norm(h)
            return torch.autograd.grad(z, h, c)[0]

    def backward_rows(self, hT, srcs, hF, ubar, a, b, tokv, read):
        """a, b [R, S] coefficient rows, tokv [R] token ids. Cotangent at norm output c = a u_X - b ubar.
        read = 'sum' (sum over valid sources) -> {layer: [R, d]}, or a LongTensor [R] of source positions."""
        R_, S = a.shape
        out = {l: [] for l in self.layers}
        nchunk = (R_ + self.B - 1) // self.B
        valid = torch.arange(SKIP, S - 1, device=self.dev)
        for ci in range(nchunk):
            sl = slice(ci * self.B, (ci + 1) * self.B)
            n = a[sl].shape[0]
            U = self.W[tokv[sl]]  # [n, d]
            c = a[sl, :, None] * U[:, None, :] - b[sl, :, None] * ubar[None]
            g = self.norm_vjp(hF, c)
            if n < self.B:
                g = torch.cat([g, g.new_zeros(self.B - n, *g.shape[1:])])
            grads = torch.autograd.grad(hT, srcs, g.to(hT.dtype), retain_graph=True)
            for l, gr in zip(self.layers, grads):
                gr = gr[:n].float()
                if isinstance(read, str):
                    out[l].append(gr[:, valid].sum(1))
                else:
                    out[l].append(gr[torch.arange(n, device=self.dev), read[sl]])
            del grads, g, c
        return {l: torch.cat(v) for l, v in out.items()}

    def rows_of(self, toks):
        new = toks[self.row[toks] < 0].unique()
        if len(new):
            n0 = len(self.ids)
            if n0 + len(new) > self.cap:
                raise RuntimeError("capacity")
            self.row[new] = torch.arange(n0, n0 + len(new), device=self.dev)
            self.ids += new.tolist()
        return self.row[toks]

    # ---------------------------------------------------------------- one window
    def window(self, ids, same_pos=True):
        S = ids.shape[0]
        valid = torch.arange(SKIP, S - 1, device=self.dev)
        hT, srcs = self.forward(ids)
        hF = hT[0].detach()
        p = self.dist(hF)
        tok, val = self.topk(p, valid)
        ubar = p @ self.W  # [S, d]
        uniq, inv = torch.unique(tok, return_inverse=True)  # inv [T, K]
        U = len(uniq)
        A = torch.zeros(U, S, device=self.dev)
        A[inv, valid[:, None].expand_as(inv)] = val
        M = torch.zeros(U, S, device=self.dev)
        M[inv, valid[:, None].expand_as(inv)] = 1.0
        if self.pool is not None:
            vals, st, ln = self.pool
            tt = tok.flatten()
            L_ = ln[tt]
            r = (torch.rand(tt.shape, device=self.dev, generator=self.gen) * L_.clamp_min(1)).long()
            w = torch.where(L_ > 0, vals[(st[tt] + r).clamp_max(len(vals) - 1)], val.flatten())
            Wd = torch.zeros(U, S, device=self.dev)
            Wd[inv, valid[:, None].expand_as(inv)] = w.view_as(val)
        else:
            Wd = A
        Z = torch.zeros_like(A)
        coef = {"p": (A, A), "a": (A, Z), "b": (M, M), "c": (M, Z), "d": (Wd, Wd)}
        same_pos = same_pos and "s" in self.D
        a = torch.cat([coef[v][0] for v in LT]); b = torch.cat([coef[v][1] for v in LT])
        tokv = uniq.repeat(len(LT))
        nrows = a.shape[0]
        if same_pos:
            T, K = tok.shape
            sa = torch.zeros(T * K, S, device=self.dev)
            pos = valid[:, None].expand(T, K).flatten()
            sa[torch.arange(T * K, device=self.dev), pos] = val.flatten()
            a = torch.cat([a, sa]); b = torch.cat([b, sa]); tokv = torch.cat([tokv, tok.flatten()])
        # all rows in one retained graph; the graph is freed when hT goes out of scope
        lt = self.backward_rows(hT, srcs, hF, ubar, a[:nrows], b[:nrows], tokv[:nrows], "sum")
        rows = self.rows_of(uniq)
        for vi, v in enumerate(LT):
            for l in self.layers:
                self.D[v][l].index_add_(0, rows, lt[l][vi * U:(vi + 1) * U])
        if same_pos:
            sp = self.backward_rows(hT, srcs, hF, ubar, a[nrows:], b[nrows:], tokv[nrows:], pos)
            rs = self.rows_of(tok.flatten())
            for l in self.layers:
                self.D["s"][l].index_add_(0, rs, sp[l])
        rr = self.rows_of(tok.flatten())
        self.N.index_add_(0, rr, torch.ones_like(rr, dtype=torch.float64))
        self.Wp.index_add_(0, rr, val.flatten().double())
        self.Wp2.index_add_(0, rr, val.flatten().double() ** 2)
        self.Wd.index_add_(0, rows, (Wd.sum(1)).double())
        del hT, srcs
        return dict(U=U, rows=nrows + (tok.numel() if same_pos else 0), T=len(valid))

    def save(self, out, meta):
        n = len(self.ids)
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        ids = torch.tensor(self.ids, dtype=torch.long)
        for v in self.D:
            dd, nn = {}, {}
            for l in self.layers:
                X = self.D[v][l][:n]
                nr = X.norm(dim=1)
                dd[l] = (X / nr.clamp_min(1e-30)[:, None]).half().cpu()
                nn[l] = nr.cpu()
            torch.save(dict(var=v, layers=self.layers, tok_ids=ids, dir=dd, norm=nn), f"{out}_{v}.pt")
        torch.save(dict(tok_ids=ids, N=self.N[:n].cpu(), Wp=self.Wp[:n].cpu(), Wp2=self.Wp2[:n].cpu(),
                        Wd=self.Wd[:n].cpu(), layers=self.layers, meta=meta), f"{out}_meta.pt")


def prepass(acc, wins, bs=32):
    """forward only: every context's top-K token p values -> per-token pools for the shuffled-weight control."""
    toks, vals = [], []
    for i in range(0, len(wins), bs):
        ids = torch.tensor(wins[i:i + bs], device=acc.dev)
        with torch.no_grad(), ActivationRecorder(acc.m.layers, at=[acc.tgt]) as rec:
            acc.m.forward(ids)
            hT = rec.activations[acc.tgt]
        S = ids.shape[1]
        valid = torch.arange(SKIP, S - 1, device=acc.dev)
        for j in range(ids.shape[0]):
            p = acc.dist(hT[j])
            t, v = acc.topk(p, valid)
            toks.append(t.flatten()); vals.append(v.flatten())
    toks = torch.cat(toks); vals = torch.cat(vals)
    o = toks.argsort(stable=True)
    toks, vals = toks[o], vals[o]
    V = acc.W.shape[0]
    ln = torch.bincount(toks, minlength=V)
    st = torch.cumsum(ln, 0) - ln
    return vals, st, ln


def sanity(acc, wins, out):
    """(1) composed cotangent (norm-output cotangent + norm VJP + model VJP) vs plain autograd of p_X(t') through
    model.unembed; (2) the log-prob identity: summed over ALL valid targets, grad log p_X - grad logit_X is the same
    vector for every X; (3) cosine between sum_c grad logit_X (all contexts) and the J-lens direction J^T (gamma*u_X)."""
    res = {}
    ids = torch.tensor(wins[0], device=acc.dev)
    S = ids.shape[0]
    valid = torch.arange(SKIP, S - 1, device=acc.dev)
    hT, srcs = acc.forward(ids)
    hF = hT[0].detach()
    p = acc.dist(hF)
    ubar = p @ acc.W
    tok, val = acc.topk(p, valid)
    # (1) direct autograd for 3 (t', X): d p_X(t') / d h_l(t) summed over valid t
    direct = []
    tests = [(int(valid[5]), int(tok[5, 0])), (int(valid[40]), int(tok[40, 3])), (int(valid[90]), int(tok[90, 9]))]
    for (tp, X) in tests:
        logits = acc.m.unembed(hT[0, tp]).float()
        pX = torch.softmax(logits, -1)[X]
        g = torch.autograd.grad(pX, [s for s in srcs], retain_graph=True)
        direct.append({l: gg[0, valid].float().sum(0) for l, gg in zip(acc.layers, g)})
    a = torch.zeros(len(tests), S, device=acc.dev)
    for i, (tp, X) in enumerate(tests):
        a[i, tp] = p[tp, X]
    comp = acc.backward_rows(hT, srcs, hF, ubar, a, a, torch.tensor([x for _, x in tests], device=acc.dev), "sum")
    res["compose_vs_direct_cos"] = {l: [float(torch.nn.functional.cosine_similarity(comp[l][i], direct[i][l], dim=0))
                                        for i in range(len(tests))] for l in acc.layers}
    res["compose_vs_direct_relerr"] = {l: [float((comp[l][i] - direct[i][l]).norm() / direct[i][l].norm())
                                           for i in range(len(tests))] for l in acc.layers}
    # (2) identity over all valid targets, 8 tokens
    Xs = torch.tensor([int(tok[10, 0]), int(tok[30, 1]), int(tok[50, 2]), int(tok[70, 4]), 1000, 5000, 20000, 60000],
                      device=acc.dev)
    Mv = torch.zeros(len(Xs), S, device=acc.dev); Mv[:, valid] = 1.0
    lp = acc.backward_rows(hT, srcs, hF, ubar, Mv, Mv, Xs, "sum")
    lg = acc.backward_rows(hT, srcs, hF, ubar, Mv, torch.zeros_like(Mv), Xs, "sum")
    ident = {}
    for l in acc.layers:
        diff = lp[l] - lg[l]  # should be the same row for every X (= - sum_c J_c^T ubar_c)
        ref = diff.mean(0, keepdim=True)
        ident[l] = dict(max_rel_dev=float(((diff - ref).norm(dim=1) / ref.norm()).max()),
                        diff_norm=float(ref.norm()), logit_dir_norm=float(lg[l].norm(dim=1).mean()))
    res["logprob_identity"] = ident
    del hT, srcs
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--cap", type=int, default=48000)
    ap.add_argument("--n-windows", type=int, default=4000)
    ap.add_argument("--budget-s", type=float, default=7200)
    ap.add_argument("--no-same-pos", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--bench-only", type=int, default=0, help="time this many windows and exit")
    ap.add_argument("--out", default=".hf27/w11/acc_2b")
    ap.add_argument("--vars", default="p,a,b,c,d", help="later-target variants to accumulate")
    args = ap.parse_args()
    LT[:] = args.vars.split(",")
    SAME[0] = not args.no_same_pos
    dev = "cuda"
    torch.manual_seed(0)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p_ in model._hf_model.parameters():
        p_.requires_grad_(False)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS[args.model]
    V = model._lm_head.weight.shape[0]
    junk = junk_mask(model.tokenizer, V)
    print(f"V {V} junk {int(junk.sum())} layers {layers} n_layers {model.n_layers}", flush=True)
    t0 = time.time()
    wins, kinds = make_windows(model.tokenizer, args.S, args.n_windows)
    print(f"windows {len(wins)} ({kinds.count('jfit')} jfit, {kinds.count('prefix')} prefix) in {time.time()-t0:.0f}s",
          flush=True)
    acc = Acc(model, layers, args.K, args.B, args.cap, dev, junk)
    if args.sanity:
        sanity(acc, wins, R / f"runs/w11/sanity_{args.model.split('-')[-1].lower()}.json")
        return
    if args.bench_only:
        for i in range(args.bench_only):
            torch.cuda.synchronize(); t1 = time.time()
            st = acc.window(torch.tensor(wins[i], device=dev), same_pos=not args.no_same_pos)
            torch.cuda.synchronize()
            print(f"win {i} {st} {time.time()-t1:.2f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        return
    t1 = time.time()
    acc.pool = prepass(acc, wins)
    print(f"prepass {len(wins)} windows {time.time()-t1:.0f}s, pool {len(acc.pool[0])} values", flush=True)
    t_start = time.time()
    nw, nctx, nrows = 0, 0, 0
    kind_n = {"jfit": 0, "prefix": 0}
    for i, w in enumerate(wins):
        try:
            st = acc.window(torch.tensor(w, device=dev), same_pos=not args.no_same_pos)
        except RuntimeError as e:
            if "capacity" not in str(e):
                raise
            print("token capacity reached, stopping", flush=True)
            break
        nw += 1; nctx += st["T"]; nrows += st["rows"]; kind_n[kinds[i]] += 1
        el = time.time() - t_start
        if nw % 20 == 0:
            print(f"[{el:.0f}s] windows {nw} contexts {nctx} rows {nrows} ctx/s {nctx/el:.1f} "
                  f"tokens {len(acc.ids)} U_last {st['U']}", flush=True)
        if el > args.budget_s:
            break
    el = time.time() - t_start
    meta = dict(model=args.model, K=args.K, S=args.S, windows=nw, contexts=nctx, rows=nrows, seconds=el,
                ctx_per_s=nctx / el, kinds=kind_n, n_tokens=len(acc.ids), same_pos=not args.no_same_pos)
    print(json.dumps(meta), flush=True)
    acc.save(args.out, meta)
    Nn = acc.N[:len(acc.ids)].cpu()
    qs = [float(x) for x in torch.quantile(Nn.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))]
    meta["N_quantiles_10_25_50_75_90_99"] = qs
    meta["N_ge"] = {k: int((Nn >= k).sum()) for k in (1, 3, 10, 30, 100, 1000)}
    (R / f"runs/w11/accum_{Path(args.out).name}.json").write_text(json.dumps(meta, indent=1))
    print("saved", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
