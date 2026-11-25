"""w11c / Experiment 5A-2, route M3 (pod): a single-matrix PROBABILITY J-lens.

Per context c = (window, valid target t'), J_c = dz(t')/dh_l summed over valid sources t <= t' (the J-lens pairing),
z = final-norm OUTPUT (logits = U z), p_c = softmax, ubar_c = U^T p_c, delta_X = u_X - ubar_c,
C_c = E_{X~p_c}[delta delta^T] = U^T diag(p_c) U - ubar_c ubar_c^T  (softmax Fisher in z-space).
    min_M sum_c tr((M - J_c)^T C_c (M - J_c))  ->  M* = B^{-1} A,  B = sum_c C_c,  A = sum_c C_c J_c   (M: [d_z, d_src])
Estimators (all of p, no top-k, no junk filter):
  A  += (1/r) sum_k delta_k (J_c^T delta_k)^T,  X_k ~ p_c (r samples, one backward row each; unbiased: E = C_c J_c)
  B  exact: U^T diag(sum_c p_c) U - sum_c ubar_c ubar_c^T   (linear in p, so summed before the [d,d] product)
  trace-normalised: A_tn, B_tn with each context weighted 1 / tr(C_c)
  identity check (C_c = I): A_I += (1/r_I) sum_k g_k (J_c^T g_k)^T, g ~ N(0, I) in z-space  ->  A_I / n = mean J_c
  rho = mean over contexts of 1 / rms(h_final) (to put the J-lens, which maps to the pre-norm residual, in z units:
        Jbar_z = rho * diag(gamma) J)
Saved: .hf27/w11/m3_<tag>.pt {A, B, A_tn, B_tn, A_I: {layer: [d, d] fp64}, n_ctx, rho, meta}
    python scripts/w11/w11c_m3b.py --budget-s 3600          (accumulate)
    python scripts/w11/w11c_m3b.py --tiny                   (exact check on 2 contexts, writes runs/w11/m3b_tiny_2b.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import LAYERS, SKIP, Acc, make_windows  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402


class M3(Acc):
    def setup(self, r, rI):
        d = self.W.shape[1]
        L = self.layers
        z = lambda: {l: torch.zeros(d, d, device=self.dev, dtype=torch.float64) for l in L}  # noqa: E731
        self.A, self.Atn, self.AI, self.A2, self.A2tn = z(), z(), z(), z(), z()
        V = self.W.shape[0]
        zv = lambda: torch.zeros(V, device=self.dev, dtype=torch.float64)  # noqa: E731
        zd = lambda: torch.zeros(d, d, device=self.dev, dtype=torch.float64)  # noqa: E731
        # W1: B = U^T diag(sum p) U - sum ubar ubar^T ; W2: B2 = U^T diag(sum p^2) U - sum (ubar w^T + w ubar^T) + sum s ubar ubar^T
        self.P, self.Ptn, self.Q, self.Qtn = zv(), zv(), zv(), zv()
        self.UU, self.UUtn, self.X2, self.X2tn = zd(), zd(), zd(), zd()
        self.r, self.rI, self.n, self.rho = r, rI, 0, 0.0
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(2)

    def pos_rows(self, hT, srcs, hF, pos, vec):
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
        hT, srcs = self.forward(ids)
        hF = hT[0].detach()
        p = self.dist(hF)[valid]  # [T, V]
        ubar = p @ self.W  # [T, d]
        X = torch.multinomial(p, self.r, replacement=True, generator=self.gen)  # [T, r]
        delta = self.W[X] - ubar[:, None]  # [T, r, d]
        posr = valid.repeat_interleave(self.r)
        Jd = self.pos_rows(hT, srcs, hF, posr, delta.reshape(-1, delta.shape[-1]))
        G = torch.randn(T * self.rI, self.W.shape[1], device=dev, generator=self.gen)
        JG = self.pos_rows(hT, srcs, hF, valid.repeat_interleave(self.rI), G)
        del hT, srcs
        un2 = (self.W ** 2).sum(1)
        trC = ((p * un2[None]).sum(1) - (ubar ** 2).sum(1)).clamp_min(1e-6)  # [T] tr C_c (W1)
        q = p ** 2
        wq = q @ self.W  # [T, d] = sum_X p_X^2 u_X
        sq = q.sum(1)  # [T]
        trC2 = ((q * un2[None]).sum(1) - 2 * (ubar * wq).sum(1) + sq * (ubar ** 2).sum(1)).clamp_min(1e-6)  # tr C2_c (W2)
        pX = p.gather(1, X).reshape(-1)  # p of each sampled token (W2 weight)
        wtn = (1.0 / trC).repeat_interleave(self.r)
        wtn2 = (pX / trC2.repeat_interleave(self.r))
        Dm = delta.reshape(-1, delta.shape[-1]).double()
        for l in self.layers:
            b = Jd[l].double()
            self.A[l] += Dm.T @ b / self.r
            self.Atn[l] += (Dm * wtn.double()[:, None]).T @ b / self.r
            self.A2[l] += (Dm * pX.double()[:, None]).T @ b / self.r
            self.A2tn[l] += (Dm * wtn2.double()[:, None]).T @ b / self.r
            self.AI[l] += G.double().T @ JG[l].double() / self.rI
        ud, wd = ubar.double(), wq.double()
        self.P += p.sum(0).double()
        self.Ptn += (p / trC[:, None]).sum(0).double()
        self.UU += ud.T @ ud
        self.UUtn += (ud / trC.double()[:, None]).T @ ud
        for (Q, X2, wt) in ((self.Q, self.X2, torch.ones_like(trC2)), (self.Qtn, self.X2tn, 1.0 / trC2)):
            wt = wt.double()
            Q += (q.double() * wt[:, None]).sum(0)
            cross = (ud * wt[:, None]).T @ wd
            X2 += -(cross + cross.T) + (ud * (sq.double() * wt)[:, None]).T @ ud
        self.rho += float((1.0 / hF[valid].float().pow(2).mean(-1).sqrt()).sum())
        self.n += T

    def B_of(self, P, UU):
        Wd = self.W.double()
        return (Wd * P[:, None]).T @ Wd - UU

    def B2_of(self, Q, X2):
        Wd = self.W.double()
        return (Wd * Q[:, None]).T @ Wd + X2

    def save(self, tag, meta):
        torch.save(dict(A={l: v.cpu() for l, v in self.A.items()}, A_tn={l: v.cpu() for l, v in self.Atn.items()},
                        A_I={l: v.cpu() for l, v in self.AI.items()}, B=self.B_of(self.P, self.UU).cpu(),
                        B_tn=self.B_of(self.Ptn, self.UUtn).cpu(),
                        A2={l: v.cpu() for l, v in self.A2.items()}, A2_tn={l: v.cpu() for l, v in self.A2tn.items()},
                        B2=self.B2_of(self.Q, self.X2).cpu(), B2_tn=self.B2_of(self.Qtn, self.X2tn).cpu(), n_ctx=self.n, rho=self.rho / self.n, layers=self.layers,
                        meta=meta), R / f".hf27/w11/m3b_{tag}.pt")


def tiny(acc, wins, out):
    """2 contexts: exact J_c (d backward rows each) and exact C_c vs the estimators."""
    ids = torch.tensor(wins[0], device=acc.dev)
    S = ids.shape[0]
    hT, srcs = acc.forward(ids)
    hF = hT[0].detach()
    pall = acc.dist(hF)
    d = acc.W.shape[1]
    l = acc.layers[-1]
    res = {}
    A_ex = torch.zeros(d, d, device=acc.dev, dtype=torch.float64); B_ex = torch.zeros_like(A_ex)
    A_est = torch.zeros_like(A_ex); J_sum = torch.zeros_like(A_ex)
    A2_ex = torch.zeros_like(A_ex); B2_ex = torch.zeros_like(A_ex); A2_est = torch.zeros_like(A_ex)
    acc2 = M3.__new__(M3); acc2.W = acc.W; acc2.dev = acc.dev; acc2.layers = acc.layers; acc2.setup(1, 1)
    nsamp = 4096
    for tp in (SKIP + 30, SKIP + 90):
        eye = torch.eye(d, device=acc.dev)
        J = acc.pos_rows(hT, srcs, hF, torch.full((d,), tp, device=acc.dev), eye)[l].double()  # rows: dz_i -> [d_src]
        p = pall[tp].double(); Wd = acc.W.double()
        ub = p @ Wd
        C = (Wd * p[:, None]).T @ Wd - torch.outer(ub, ub)
        A_ex += C @ J; B_ex += C; J_sum += J
        q = p ** 2
        C2 = (Wd * q[:, None]).T @ Wd - torch.outer(ub, q @ Wd) - torch.outer(q @ Wd, ub) + q.sum() * torch.outer(ub, ub)
        A2_ex += C2 @ J; B2_ex += C2
        # accumulator formula for B2 (as in window())
        wq = q @ Wd
        acc2.Q += q; acc2.X2 += -(torch.outer(ub, wq) + torch.outer(wq, ub)) + q.sum() * torch.outer(ub, ub)
        X = torch.multinomial(pall[tp], nsamp, replacement=True, generator=acc.gen)
        delta = acc.W[X] - pall[tp] @ acc.W
        b = acc.pos_rows(hT, srcs, hF, torch.full((nsamp,), tp, device=acc.dev), delta)[l].double()
        A_est += delta.double().T @ b / nsamp
        A2_est += (delta.double() * pall[tp].double()[X][:, None]).T @ b / nsamp
        # identity: C = I gives sum J_c
    rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
    res["A_est_vs_exact_relF"] = rel(A_est, A_ex)
    res["A_est_vs_exact_cos"] = float((A_est * A_ex).sum() / A_est.norm() / A_ex.norm())
    # B formula (linear in p) vs explicit sum of C_c
    P = pall[SKIP + 30].double() + pall[SKIP + 90].double()
    ubs = [pall[t].double() @ acc.W.double() for t in (SKIP + 30, SKIP + 90)]
    B_f = (acc.W.double() * P[:, None]).T @ acc.W.double() - sum(torch.outer(u, u) for u in ubs)
    res["B_formula_vs_exact_relF"] = rel(B_f, B_ex)
    ev = torch.linalg.eigvalsh(B_ex)
    res["B_tiny_eig_min_max"] = [float(ev.min()), float(ev.max())]
    lam = 1e-3 * float(ev.max())
    M_ex = torch.linalg.solve(B_ex + lam * torch.eye(d, device=acc.dev, dtype=torch.float64), A_ex)
    M_es = torch.linalg.solve(B_ex + lam * torch.eye(d, device=acc.dev, dtype=torch.float64), A_est)
    res["Mstar_est_vs_exact_relF(ridge 1e-3*eigmax)"] = rel(M_es, M_ex)
    res["C=I check: M = (sum I)^-1 sum J_c = mean J_c, relF to mean J"] = rel(
        torch.linalg.solve(2 * torch.eye(d, device=acc.dev, dtype=torch.float64), J_sum), J_sum / 2)
    res["W2_A_est_vs_exact_relF"] = rel(A2_est, A2_ex)
    res["W2_A_est_vs_exact_cos"] = float((A2_est * A2_ex).sum() / A2_est.norm() / A2_ex.norm())
    res["W2_B_formula_vs_exact_relF"] = rel(acc2.B2_of(acc2.Q, acc2.X2), B2_ex)
    ev2 = torch.linalg.eigvalsh(B2_ex)
    res["W2_B_tiny_eig_min_max"] = [float(ev2.min()), float(ev2.max())]
    lam2 = 1e-3 * float(ev2.max())
    I = torch.eye(d, device=acc.dev, dtype=torch.float64)
    res["W2_Mstar_est_vs_exact_relF(ridge 1e-3*eigmax)"] = rel(torch.linalg.solve(B2_ex + lam2 * I, A2_est),
                                                              torch.linalg.solve(B2_ex + lam2 * I, A2_ex))
    for f in (1e-2, 1e-1):
        res[f"W1_Mstar_relF ridge {f}*eigmax"] = rel(torch.linalg.solve(B_ex + f * float(ev.max()) * I, A_est),
                                                    torch.linalg.solve(B_ex + f * float(ev.max()) * I, A_ex))
        res[f"W2_Mstar_relF ridge {f}*eigmax"] = rel(torch.linalg.solve(B2_ex + f * float(ev2.max()) * I, A2_est),
                                                    torch.linalg.solve(B2_ex + f * float(ev2.max()) * I, A2_ex))
    res["layer"] = l
    res["samples_per_context"] = nsamp
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tag", default="2b")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--rI", type=int, default=4)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=3000)
    ap.add_argument("--budget-s", type=float, default=3600)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p_ in model._hf_model.parameters():
        p_.requires_grad_(False)
    L = LAYERS[args.model]
    V = model._lm_head.weight.shape[0]
    wins, kinds = make_windows(model.tokenizer, 128, args.n_windows)
    acc = M3(model, L, 10, args.B, 1, dev, torch.zeros(V, dtype=torch.bool))
    acc.D = {}
    acc.setup(args.r, args.rI)
    if args.tiny:
        acc.layers = [L[-1]]
        tiny(acc, wins, R / f"runs/w11/m3b_tiny_{args.tag}.json")
        return
    t0 = time.time()
    nw = 0
    for w in wins:
        acc.window(torch.tensor(w, device=dev))
        nw += 1
        el = time.time() - t0
        if nw % 20 == 0:
            print(f"[{el:.0f}s] windows {nw} contexts {acc.n} ctx/s {acc.n/el:.1f}", flush=True)
        if el > args.budget_s:
            break
    meta = dict(model=args.model, layers=L, r=args.r, rI=args.rI, windows=nw, contexts=acc.n, seconds=time.time() - t0,
                jfit=kinds[:nw].count("jfit"), prefix=kinds[:nw].count("prefix"))
    acc.save(args.tag, meta)
    (R / f"runs/w11/m3b_{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print("saved", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
