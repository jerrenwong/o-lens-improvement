"""w11b / Experiment 5A-2, estimator M2 (pod): per-token mixture over CLUSTER Jacobians, forward passes only.

Experts J_k (plain rules, lrp none; scripts/w11/m2_fit.sh, routers K = 8 and 64 at layers 15/18/21) are the
position-weighted means over source positions s in cluster k of  sum_{valid t' >= s} dh_final(t') / dh_l(s)
(the J-lens estimator, bucketed by the router of h_l(s)). With the J-lens readout (gamma * u_X, no norm Jacobian):
   D_X = sum_s J_{k(s)}^T G [ pbar_X(s) u_X - (1/n_s) sum_{t' >= s} p_X(t') ubar(t') ]
       = sum_k w_Xk J_k^T G u_X  -  T2_X,         G = diag(gamma)
   pbar_X(s) = (1/n_s) sum_{valid t' >= s} p_X(t')  (n_s = number of valid targets >= s: the per-pair Jacobian is
   approximated by the source's cluster Jacobian, the same for every later target)
   w_Xk = sum_{s in k} pbar_X(s);   T2_X = sum_t' p_X(t') q(t'),  q(t') = sum_k c_k(t') J_k^T G ubar(t'),
   c_k(t') = sum_{valid s <= t', k(s) = k} 1/n_s;   W_X = sum_k w_Xk.
Uniform cluster weights (w_Xk proportional to cluster size, no ubar term) give back the pooled J-lens.
Shuffled control: p (and pbar) of each window replaced by the previous window's (global cyclic permutation).
All of p over the whole vocabulary, no junk filter. Saved: .hf27/w11/m2_<tag>_K<K>.pt
    {w: {l: [V, K]}, T2: {l: [V, d] bf16}, wsh, T2sh, W, n_win, n_src}
    python scripts/w11/w11b_m2.py --n-windows 8000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import SKIP, make_windows  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402


def head_gamma(model):
    norm, head = model._final_norm, model._lm_head
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    p = next(norm.parameters())
    ones = torch.ones(1, head.weight.shape[1], dtype=torch.float32, device=p.device)
    return (norm(ones).float()[0] * (1.0 + eps) ** 0.5).to(head.weight.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tag", default="2b")
    ap.add_argument("--layers", default="15,18,21")
    ap.add_argument("--Ks", default="8,64")
    ap.add_argument("--n-windows", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    m = get_hf_model(args.model, attn_implementation="sdpa")
    L = [int(x) for x in args.layers.split(",")]
    Ks = [int(x) for x in args.Ks.split(",")]
    Wu = m._lm_head.weight.float()
    V, d = Wu.shape
    gam = head_gamma(m).float()
    import copy
    norm = copy.deepcopy(m._final_norm).float()
    B = R / ".hf27/w11/m2fit"
    routers = {K: ActivationRouterCollection.load(str(B / f"router_K{K}.pt")) for K in Ks}
    for K in Ks:
        for lr in routers[K].layer_routers_L_dict.values():
            for k_, v_ in vars(lr).items():
                if torch.is_tensor(v_):
                    setattr(lr, k_, v_.to(dev))
    EX = {}
    for K in Ks:
        st = torch.load(B / f"experts_K{K}.pt", map_location="cpu", weights_only=False)
        EX[K] = {l: st["experts"][l].to(dev).float() for l in L}  # [K, F(target), N(source)]
    wins, kinds = make_windows(m.tokenizer, 128, args.n_windows)
    S = 128
    valid = torch.arange(SKIP, S - 1, device=dev)
    T = len(valid)
    ns = torch.arange(T, 0, -1, device=dev).float()  # valid targets >= s
    acc = {K: dict(w={l: torch.zeros(V, K, device=dev) for l in L}, T2={l: torch.zeros(V, d, device=dev) for l in L},
                   wsh={l: torch.zeros(V, K, device=dev) for l in L}, T2sh={l: torch.zeros(V, d, device=dev) for l in L})
           for K in Ks}
    Wv = torch.zeros(V, device=dev, dtype=torch.float64)
    prev = None
    t0 = time.time()
    nwin = 0
    tri = torch.tril(torch.ones(T, T, device=dev))  # [t', s]: s <= t'
    with torch.no_grad():
        for i in range(0, len(wins), args.bs):
            ids = torch.tensor(wins[i:i + args.bs], device=dev)
            with ActivationRecorder(m.layers, at=[*L, m.n_layers - 1]) as rec:
                m.forward(ids)
            for j in range(ids.shape[0]):
                hF = rec.activations[m.n_layers - 1][j]
                p = torch.softmax(norm(hF.float()) @ Wu.T, -1)[valid]  # [T, V]
                pbar = p.flip(0).cumsum(0).flip(0) / ns[:, None]  # [T(s), V]
                ubG = (p @ Wu) * gam  # [T(t'), d]
                for l in L:
                    a = rec.activations[l][j][valid].float()
                    for K in Ks:
                        k = routers[K].assign(a, l)
                        oh = torch.nn.functional.one_hot(k, K).float()  # [T(s), K]
                        c = tri @ (oh / ns[:, None])  # [T(t'), K]
                        Y = torch.einsum("tf,kfn->ktn", ubG, EX[K][l])  # [K, T, N]
                        q = torch.einsum("tk,ktn->tn", c, Y)
                        A = acc[K]
                        A["w"][l].addmm_(pbar.T, oh)
                        A["T2"][l].addmm_(p.T, q)
                        if prev is not None:
                            A["wsh"][l].addmm_(prev[1].T, oh)
                            A["T2sh"][l].addmm_(prev[0].T, q)
                Wv += pbar.sum(0).double()
                prev = (p, pbar)
                nwin += 1
            if (i // args.bs) % 25 == 0:
                print(f"[{time.time()-t0:.0f}s] windows {nwin}", flush=True)
    meta = dict(model=args.model, layers=L, Ks=Ks, windows=nwin, sources=nwin * T, seconds=time.time() - t0,
                jfit=kinds[:nwin].count("jfit"), prefix=kinds[:nwin].count("prefix"))
    for K in Ks:
        A = acc[K]
        torch.save(dict(w={l: v.cpu() for l, v in A["w"].items()}, wsh={l: v.cpu() for l, v in A["wsh"].items()},
                        T2={l: v.to(torch.bfloat16).cpu() for l, v in A["T2"].items()},
                        T2sh={l: v.to(torch.bfloat16).cpu() for l, v in A["T2sh"].items()},
                        W=Wv.cpu(), meta=meta), R / f".hf27/w11/m2_{args.tag}_K{K}.pt")
    (R / f"runs/w11/m2_{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print("saved", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
