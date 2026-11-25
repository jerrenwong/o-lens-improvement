"""w11b / Experiment 5A-2, estimator M1 (pod): probability-gradient directions with ALL of p (no top-K cut, no junk
filter in the fit), for EVERY vocabulary token, dense [V, d] tables at a few readout layers.

Target (same (source, later-target) pairing as the J-lens fit, valid positions 16..S-2, context c = (window, t')):
    D_X = sum_c p_X(c) J_c^T (u_X - ubar_c) = T1_X - T2_X,   W_X = sum_c p_X(c)
J_c^T = Jacobian of the final-norm OUTPUT z(t') w.r.t. h_l(t), summed over valid sources t <= t' (autograd through
the real final norm; logits = U z).
  T2 exact: v_c = J_c^T ubar_c, one backward row per context; T2 += p_c (outer) v_c.
  T1 sketched: r Gaussian probes g_ck in z-space per context (one backward row each):
      J_c^T u_X ~= (1/r) sum_k (g_ck . u_X) J_c^T g_ck  (unbiased);   T1 += (p_c(X) (u_X.g_ck)/r) (J_c^T g_ck)
  PU = pure sketch for all tokens.
  HY = hybrid: exact J_c^T u_X for the top-K tokens of each context (K = 10, no junk filter; one backward row per
       distinct top token per window, as the earlier w11 'a' rows) + sketch for the tail (top-K entries masked).
  SH = shuffled-weight control: p of each context replaced by p of the context at the same position in the PREVIOUS
       window (a global cyclic permutation of contexts, so every token keeps its own weight distribution), pure sketch.
  B  = sum_c (1/r) sum_k (J_c^T g_ck) g_ck^T  [d_src, d_z]: the uniform-weight control (the J-lens with this
       sketch and these contexts): D_unif_X = B u_X.
Split-half: for a token subset (fit+dev candidate ids + 3000 random ids), T1/T2/W accumulated separately on even /
odd windows -> cosine between the two half estimates of D_X.
Saved (bf16): .hf27/w11/m1_<tag>_{Dpu,Dhy,Apu,Ahy,Dsh}.pt = {layer: [V, d]} with D = T1 - T2, A = T1 (weighting
only); meta .hf27/w11/m1_<tag>_meta.pt (W, W2, Wsh, B, n_ctx); split-half runs/w11/m1_<tag>_splithalf.pt.
    python scripts/w11/w11b_accum.py --model Qwen/Qwen3.5-2B --layers 15,18,21 --r 16 --budget-s 7200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import SKIP, Acc, make_windows  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

sys.path.insert(0, str(R / "src"))
from olens.readout import Bench  # noqa: E402


class M1(Acc):
    def setup(self, r, sub_ids):
        V, d = self.W.shape
        dev = self.dev
        self.r = r
        L = self.layers
        self.T = {k: {l: torch.zeros(V, d, device=dev) for l in L} for k in ("PU", "HY", "T2", "SH1", "SH2")}
        self.Bm = {l: torch.zeros(d, d, device=dev) for l in L}
        self.Wv = torch.zeros(V, device=dev, dtype=torch.float64)
        self.W2v = torch.zeros(V, device=dev, dtype=torch.float64)
        self.Wsh = torch.zeros(V, device=dev, dtype=torch.float64)
        self.sub = sub_ids.to(dev)
        self.subpos = torch.full((V,), -1, dtype=torch.long, device=dev)
        self.subpos[self.sub] = torch.arange(len(self.sub), device=dev)
        n = len(self.sub)
        self.H = {k: [{l: torch.zeros(n, d, device=dev) for l in L} for _ in range(2)] for k in ("PU", "HY", "T2")}
        self.Wh = [torch.zeros(n, device=dev, dtype=torch.float64) for _ in range(2)]
        self.prev = None
        self.nctx = 0
        self.nwin = 0
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(1)

    def pos_rows(self, hT, srcs, hF, pos, vec):
        """rows with cotangent vec[i] (z-space) at position pos[i] only -> {layer: [n, d]} summed over valid sources"""
        n = len(pos)
        S = hF.shape[0]
        valid = torch.arange(SKIP, S - 1, device=self.dev)
        out = {l: [] for l in self.layers}
        for i in range(0, n, self.B):
            m = min(self.B, n - i)
            c = torch.zeros(m, S, vec.shape[1], device=self.dev)
            c[torch.arange(m, device=self.dev), pos[i:i + m]] = vec[i:i + m]
            g = self.norm_vjp(hF, c)
            if m < self.B:
                g = torch.cat([g, g.new_zeros(self.B - m, *g.shape[1:])])
            grads = torch.autograd.grad(hT, srcs, g.to(hT.dtype), retain_graph=True)
            for l, gr in zip(self.layers, grads):
                out[l].append(gr[:m, valid].float().sum(1))
            del grads, g, c
        return {l: torch.cat(v) for l, v in out.items()}

    def window(self, ids):
        S = ids.shape[0]
        dev = self.dev
        valid = torch.arange(SKIP, S - 1, device=dev)
        T = len(valid)
        r = self.r
        hT, srcs = self.forward(ids)
        hF = hT[0].detach()
        p = self.dist(hF)  # [S, V]
        ubar = p @ self.W
        pv = p[valid]  # [T, V]
        val, tok = pv.topk(self.K, dim=-1)  # no junk filter
        # (1) probes, (2) v_c rows
        G = torch.randn(T * r, self.W.shape[1], device=dev, generator=self.gen)
        posr = valid.repeat_interleave(r)
        JG = self.pos_rows(hT, srcs, hF, posr, G)
        Jv = self.pos_rows(hT, srcs, hF, valid, ubar[valid])
        # (3) exact top-K rows (one per distinct top token)
        uniq, inv = torch.unique(tok, return_inverse=True)
        U = len(uniq)
        A = torch.zeros(U, S, device=dev)
        A[inv, valid[:, None].expand_as(inv)] = val
        Ex = self.backward_rows(hT, srcs, hF, ubar, A, torch.zeros_like(A), uniq, "sum")
        del hT, srcs
        # coefficients [V, T*r]
        UG = self.W @ G.T  # [V, T*r]
        prep = pv.T.repeat_interleave(r, dim=1)
        Cpu = UG * prep / r
        mask = torch.zeros(T, self.W.shape[0], device=dev)
        mask.scatter_(1, tok, 1.0)
        Chy = Cpu * (1.0 - mask.T.repeat_interleave(r, dim=1))
        del prep, mask
        Csh = None
        if self.prev is not None:
            Csh = UG * self.prev.T.repeat_interleave(r, dim=1) / r
        h = self.nwin % 2
        sp = self.subpos[uniq]
        ins = sp >= 0
        for l in self.layers:
            self.T["PU"][l].addmm_(Cpu, JG[l])
            self.T["HY"][l].addmm_(Chy, JG[l])
            self.T["HY"][l].index_add_(0, uniq, Ex[l])
            self.T["T2"][l].addmm_(pv.T, Jv[l])
            if Csh is not None:
                self.T["SH1"][l].addmm_(Csh, JG[l])
                self.T["SH2"][l].addmm_(self.prev.T, Jv[l])
            self.Bm[l].addmm_(JG[l].T, G, alpha=1.0 / r)
            self.H["PU"][h][l].addmm_(Cpu[self.sub], JG[l])
            self.H["HY"][h][l].addmm_(Chy[self.sub], JG[l])
            self.H["HY"][h][l].index_add_(0, sp[ins], Ex[l][ins])
            self.H["T2"][h][l].addmm_(pv[:, self.sub].T, Jv[l])
        self.Wv += pv.sum(0).double(); self.W2v += (pv.double() ** 2).sum(0)
        self.Wh[h] += pv[:, self.sub].sum(0).double()
        if self.prev is not None:
            self.Wsh += self.prev.sum(0).double()
        self.prev = pv.clone()
        self.nctx += T; self.nwin += 1
        return dict(U=U, rows=T * r + T + U, T=T)

    def save(self, tag, meta):
        L = self.layers
        combos = {"Dpu": ("PU", "T2"), "Dhy": ("HY", "T2"), "Apu": ("PU", None), "Ahy": ("HY", None), "Dsh": ("SH1", "SH2")}
        for name, (a, b) in combos.items():
            out = {}
            for l in L:
                X = self.T[a][l] if b is None else self.T[a][l] - self.T[b][l]
                out[l] = X.to(torch.bfloat16).cpu()
            torch.save(out, R / f".hf27/w11/m1_{tag}_{name}.pt")
        torch.save(dict(W=self.Wv.cpu(), W2=self.W2v.cpu(), Wsh=self.Wsh.cpu(), B={l: v.cpu() for l, v in self.Bm.items()},
                        n_ctx=self.nctx, n_win=self.nwin, layers=L, r=self.r, meta=meta), R / f".hf27/w11/m1_{tag}_meta.pt")
        # split-half cosines
        sh = dict(sub=self.sub.cpu(), Wh=[w.cpu() for w in self.Wh], W=self.Wv[self.sub].cpu(), W2=self.W2v[self.sub].cpu())
        for name, (a, b) in {"Dpu": ("PU", "T2"), "Dhy": ("HY", "T2"), "Apu": ("PU", None), "Ahy": ("HY", None)}.items():
            sh[name] = {}
            for l in L:
                X = [self.H[a][k][l] - (0 if b is None else self.H[b][k][l]) for k in range(2)]
                sh[name][l] = torch.nn.functional.cosine_similarity(X[0], X[1], dim=1).cpu()
        torch.save(sh, R / f"runs/w11/m1_{tag}_splithalf.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--layers", default="15,18,21")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=3000)
    ap.add_argument("--budget-s", type=float, default=7200)
    ap.add_argument("--bench-only", type=int, default=0)
    ap.add_argument("--tag", default="2b")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    torch.manual_seed(0)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p_ in model._hf_model.parameters():
        p_.requires_grad_(False)
    layers = [int(x) for x in args.layers.split(",")]
    V = model._lm_head.weight.shape[0]
    wins, kinds = make_windows(model.tokenizer, 128, args.n_windows)
    bench = Bench.load(R / f"artifacts/bench/qwen3.5-{args.tag}.pt")
    cand = sorted({t for c in bench.candidates for ids in c.values() for t in ids})
    g = torch.Generator().manual_seed(0)
    rnd = torch.randperm(V, generator=g)[:3000].tolist()
    sub = torch.tensor(sorted(set(cand) | set(rnd)), dtype=torch.long)
    print(f"V {V} layers {layers} windows {len(wins)} subset {len(sub)}", flush=True)
    acc = M1(model, layers, args.K, args.B, 1, dev, torch.zeros(V, dtype=torch.bool))
    acc.D = {}
    acc.setup(args.r, sub)
    t0 = time.time()
    for i, w in enumerate(wins):
        torch.cuda.synchronize(); t1 = time.time()
        st = acc.window(torch.tensor(w, device=dev))
        el = time.time() - t0
        if args.bench_only:
            torch.cuda.synchronize()
            print(f"win {i} {st} {time.time()-t1:.2f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            if i + 1 >= args.bench_only:
                return
        elif (i + 1) % 20 == 0:
            print(f"[{el:.0f}s] windows {i+1} contexts {acc.nctx} ctx/s {acc.nctx/el:.1f}", flush=True)
        if el > args.budget_s:
            break
    el = time.time() - t0
    meta = dict(model=args.model, layers=layers, r=args.r, K=args.K, windows=acc.nwin, contexts=acc.nctx, seconds=el,
                ctx_per_s=acc.nctx / el, jfit=kinds[:acc.nwin].count("jfit"), prefix=kinds[:acc.nwin].count("prefix"))
    print(json.dumps(meta), flush=True)
    acc.save(args.tag, meta)
    (R / f"runs/w11/m1_{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
