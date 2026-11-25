"""w11c / route M3, DETERMINISTIC p^2 (W2x) and p^3 (W3x) estimators (pod).
Per context c (J-lens pairing, z-space, real final norm), exact top-K tokens of p_c (K = 10, no junk filter, no
sampling), delta_X = u_X - ubar_c with ubar_c from the FULL p:
   A2x += sum_{X in topK} p_X^2 delta_X (J_c^T delta_X)^T        A3x: p_X^3 instead of p_X^2
   B2x  = sum_c sum_{ALL X} p_X^2 delta_X delta_X^T  (exact, linear in p^2:  U^T diag(q) U - ubar w^T - w ubar^T + s ubar ubar^T,
          q = p^2, w = U^T q, s = sum q)                          B3x: q = p^3
   trace-normalised: each context divided by tr C2_c (tr C3_c)
Same pass, for split-half and for the W2 sampled-vs-exact comparison: the sampled W2 estimator (r = 8 tokens X ~ p_c,
weights p_X). Everything is accumulated separately on even / odd windows (halves; full = sum).
Per context: truncation share 1 - sum_topK p^2 / sum_all p^2 (and p^3), and the K needed for 99% of sum p^2.
Saved .hf27/w11/m3x_<tag>.pt; stats runs/w11/m3x_trunc_<tag>.json
    python scripts/w11/w11c_m3x.py --budget-s 2700
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import LAYERS, SKIP, make_windows  # noqa: E402
from w11c_m3b import M3  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402


class M3x(M3):
    def setup2(self, K, r):
        d = self.W.shape[1]; V = self.W.shape[0]; L = self.layers
        z = lambda: [{l: torch.zeros(d, d, device=self.dev, dtype=torch.float64) for l in L} for _ in range(2)]  # noqa
        self.Ak = {k: z() for k in ("A2x", "A3x", "A2x_tn", "A3x_tn", "A2s")}
        self.Q = {k: [torch.zeros(V, device=self.dev, dtype=torch.float64) for _ in range(2)] for k in ("2", "3", "2tn", "3tn")}
        self.X = {k: [torch.zeros(d, d, device=self.dev, dtype=torch.float64) for _ in range(2)] for k in ("2", "3", "2tn", "3tn")}
        self.K2, self.r2 = K, r
        self.trunc, self.k99, self.trunc3 = [], [], []
        self.nwin = 0; self.n = 0

    def window2(self, ids):
        dev = self.dev
        S = ids.shape[0]
        valid = torch.arange(SKIP, S - 1, device=dev)
        T = len(valid)
        h = self.nwin % 2
        hT, srcs = self.forward(ids)
        hF = hT[0].detach()
        p = self.dist(hF)[valid]
        ub = p @ self.W
        val, tok = p.topk(self.K2, dim=-1)
        dK = self.W[tok] - ub[:, None]  # [T, K, d]
        bK = self.pos_rows(hT, srcs, hF, valid.repeat_interleave(self.K2), dK.reshape(-1, dK.shape[-1]))
        X = torch.multinomial(p, self.r2, replacement=True, generator=self.gen)
        dS = self.W[X] - ub[:, None]
        bS = self.pos_rows(hT, srcs, hF, valid.repeat_interleave(self.r2), dS.reshape(-1, dS.shape[-1]))
        del hT, srcs
        un2 = (self.W ** 2).sum(1)
        tr = {}
        for e in (2, 3):
            q = p ** e
            wq = q @ self.W
            sq = q.sum(1)
            tr[e] = ((q * un2[None]).sum(1) - 2 * (ub * wq).sum(1) + sq * (ub ** 2).sum(1)).clamp_min(1e-9)
            for suf, wt in (("", torch.ones_like(sq)), ("tn", 1.0 / tr[e])):
                wt = wt.double()
                self.Q[f"{e}{suf}"][h] += (q.double() * wt[:, None]).sum(0)
                cross = (ub.double() * wt[:, None]).T @ wq.double()
                self.X[f"{e}{suf}"][h] += -(cross + cross.T) + (ub.double() * (sq.double() * wt)[:, None]).T @ ub.double()
        DK = dK.reshape(-1, dK.shape[-1]).double()
        DS = dS.reshape(-1, dS.shape[-1]).double()
        v2 = (val ** 2).reshape(-1).double(); v3 = (val ** 3).reshape(-1).double()
        t2 = tr[2].repeat_interleave(self.K2).double(); t3 = tr[3].repeat_interleave(self.K2).double()
        pS = p.gather(1, X).reshape(-1).double()
        for l in self.layers:
            b = bK[l].double(); bs = bS[l].double()
            self.Ak["A2x"][h][l] += (DK * v2[:, None]).T @ b
            self.Ak["A3x"][h][l] += (DK * v3[:, None]).T @ b
            self.Ak["A2x_tn"][h][l] += (DK * (v2 / t2)[:, None]).T @ b
            self.Ak["A3x_tn"][h][l] += (DK * (v3 / t3)[:, None]).T @ b
            self.Ak["A2s"][h][l] += (DS * pS[:, None]).T @ bs / self.r2
        p2 = p.double() ** 2
        tot2 = p2.sum(1)
        self.trunc.append((1 - (val.double() ** 2).sum(1) / tot2).cpu())
        self.trunc3.append((1 - (val.double() ** 3).sum(1) / (p.double() ** 3).sum(1)).cpu())
        cs = (p2.sort(dim=1, descending=True).values[:, :512].cumsum(1) / tot2[:, None])
        self.k99.append(((cs < 0.99).sum(1) + 1).cpu())
        self.nwin += 1; self.n += T

    def B_e(self, e, suf, h):
        Wd = self.W.double()
        return (Wd * self.Q[f"{e}{suf}"][h][:, None]).T @ Wd + self.X[f"{e}{suf}"][h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tag", default="2b")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=2700)
    args = ap.parse_args()
    dev = "cuda"
    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p_ in model._hf_model.parameters():
        p_.requires_grad_(False)
    L = LAYERS[args.model]
    V = model._lm_head.weight.shape[0]
    acc = M3x(model, L, 10, 64, 1, dev, torch.zeros(V, dtype=torch.bool))
    acc.D = {}
    acc.setup(1, 1)
    acc.setup2(args.K, args.r)
    wins, kinds = make_windows(model.tokenizer, 128, 3000)
    t0 = time.time()
    for w in wins:
        acc.window2(torch.tensor(w, device=dev))
        el = time.time() - t0
        if acc.nwin % 20 == 0:
            print(f"[{el:.0f}s] windows {acc.nwin} contexts {acc.n} ctx/s {acc.n/el:.1f}", flush=True)
        if el > args.budget_s:
            break
    out = dict(layers=L, n_ctx=acc.n, n_win=acc.nwin, meta=dict(K=args.K, r=args.r, windows=acc.nwin, contexts=acc.n,
                                                                    seconds=time.time() - t0))
    for k, v in acc.Ak.items():
        out[k] = [{l: x.cpu() for l, x in half.items()} for half in v]
    for e in (2, 3):
        for suf in ("", "tn"):
            out[f"B{e}x" + ("_tn" if suf else "")] = [acc.B_e(e, suf, h).cpu() for h in range(2)]
    torch.save(out, R / f".hf27/w11/m3x_{args.tag}.pt")
    tr = torch.cat(acc.trunc); tr3 = torch.cat(acc.trunc3); k99 = torch.cat(acc.k99).double()
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], dtype=torch.float64)
    st = dict(n_ctx=acc.n, K=args.K,
              trunc_share_p2_quantiles_10_25_50_75_90_99=[round(float(x), 4) for x in torch.quantile(tr, qs)],
              trunc_share_p2_mean=float(tr.mean()),
              trunc_share_p3_quantiles=[round(float(x), 5) for x in torch.quantile(tr3, qs)],
              K_for_99pct_of_sum_p2_quantiles=[float(x) for x in torch.quantile(k99, qs)])
    (R / f"runs/w11/m3x_trunc_{args.tag}.json").write_text(json.dumps(st, indent=1))
    print("saved", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
