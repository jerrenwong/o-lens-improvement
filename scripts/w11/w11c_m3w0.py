"""w11c / route M3, uncentred weighting W0 (pod): C_c^(0) = sum_X p_X u_X u_X^T = C_c + ubar_c ubar_c^T, so
    A0 = A1 + sum_c ubar_c (J_c^T ubar_c)^T,   B0 = B1 + sum_c ubar_c ubar_c^T.
The extra term is deterministic given the contexts, so it is computed EXACTLY in a second pass over the same windows
the M3 pass used (first n_win of make_windows; one backward row per context: cotangent ubar_c at t'), and added to the
sampled A1 of .hf27/w11/m3b_<tag>.pt. Also records per context tr(C_c), tr(C2_c), |ubar_c|^2, max p (how much weight
confident contexts carry under W1 / W2 / W0).
    python scripts/w11/w11c_m3w0.py            -> .hf27/w11/m3b_<tag>_w0.pt, runs/w11/m3_ctxweights_<tag>.json
    python scripts/w11/w11c_m3w0.py --tiny     -> runs/w11/m3w0_tiny_<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import LAYERS, SKIP, make_windows  # noqa: E402
from w11c_m3b import M3  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tag", default="2b")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p_ in model._hf_model.parameters():
        p_.requires_grad_(False)
    L = LAYERS[args.model]
    V = model._lm_head.weight.shape[0]
    acc = M3(model, L, 10, 64, 1, dev, torch.zeros(V, dtype=torch.bool))
    acc.D = {}
    acc.setup(1, 1)
    if args.tiny:
        wins, _ = make_windows(model.tokenizer, 128, 4)
        l = L[-1]; acc.layers = [l]
        ids = torch.tensor(wins[0], device=dev)
        hT, srcs = acc.forward(ids)
        hF = hT[0].detach(); pall = acc.dist(hF); d = acc.W.shape[1]
        Wd = acc.W.double()
        A0_ex = torch.zeros(d, d, device=dev, dtype=torch.float64); A0_est = torch.zeros_like(A0_ex)
        for tp in (SKIP + 30, SKIP + 90):
            J = acc.pos_rows(hT, srcs, hF, torch.full((d,), tp, device=dev), torch.eye(d, device=dev))[l].double()
            p = pall[tp].double(); ub = p @ Wd
            C0 = (Wd * p[:, None]).T @ Wd
            A0_ex += C0 @ J
            X = torch.multinomial(pall[tp], 4096, replacement=True)
            delta = acc.W[X] - pall[tp] @ acc.W
            b = acc.pos_rows(hT, srcs, hF, torch.full((4096,), tp, device=dev), delta)[l].double()
            Ju = acc.pos_rows(hT, srcs, hF, torch.tensor([tp], device=dev), (pall[tp] @ acc.W)[None])[l].double()[0]
            A0_est += delta.double().T @ b / 4096 + torch.outer(ub, Ju)
        res = dict(W0_A_est_vs_exact_relF=float((A0_est - A0_ex).norm() / A0_ex.norm()),
                   W0_A_est_vs_exact_cos=float((A0_est * A0_ex).sum() / A0_est.norm() / A0_ex.norm()), layer=l,
                   note="A0 estimate = sampled A1 (4096 samples/context) + exact ubar (J^T ubar)^T")
        (R / f"runs/w11/m3w0_tiny_{args.tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    st = torch.load(R / f".hf27/w11/m3b_{args.tag}.pt", map_location="cpu")
    nwin = st["meta"]["windows"]
    wins, _ = make_windows(model.tokenizer, 128, 3000)
    d = acc.W.shape[1]
    E = {l: torch.zeros(d, d, device=dev, dtype=torch.float64) for l in L}
    UU = torch.zeros(d, d, device=dev, dtype=torch.float64)
    stats = []
    un2 = (acc.W ** 2).sum(1)
    for i in range(nwin):
        ids = torch.tensor(wins[i], device=dev)
        valid = torch.arange(SKIP, ids.shape[0] - 1, device=dev)
        hT, srcs = acc.forward(ids)
        hF = hT[0].detach()
        p = acc.dist(hF)[valid]
        ub = p @ acc.W
        Ju = acc.pos_rows(hT, srcs, hF, valid, ub)
        del hT, srcs
        for l in L:
            E[l] += ub.double().T @ Ju[l].double()
        UU += ub.double().T @ ub.double()
        q = p ** 2; wq = q @ acc.W
        trC = (p * un2[None]).sum(1) - (ub ** 2).sum(1)
        trC2 = (q * un2[None]).sum(1) - 2 * (ub * wq).sum(1) + q.sum(1) * (ub ** 2).sum(1)
        stats.append(torch.stack([trC, trC2, (ub ** 2).sum(1), p.max(1).values], 1).cpu())
        if (i + 1) % 100 == 0:
            print(f"windows {i+1}/{nwin}", flush=True)
    A0 = {l: st["A"][l] + E[l].cpu() for l in L}
    B0 = st["B"] + UU.cpu()
    torch.save(dict(A0=A0, B0=B0, n_win=nwin), R / f".hf27/w11/m3b_{args.tag}_w0.pt")
    S = torch.cat(stats).double()  # [n, 4]: trC, trC2, |ubar|^2, maxp
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    tot = {k: S[:, j].sum() for j, k in enumerate(["trC", "trC2", "ub2"])}
    conf = S[:, 3] > 0.9
    out = dict(n_ctx=int(S.shape[0]),
               quantiles_10_25_50_75_90={k: [float(x) for x in torch.quantile(S[:, j], qs)] for j, k in
                                         enumerate(["trC (W1 weight)", "trC2 (W2 weight)", "|ubar|^2", "max p"])},
               share_confident_ctx_maxp_gt_0_9=float(conf.double().mean()),
               weight_share_of_confident_ctx={"W1 (trC)": float(S[conf, 0].sum() / tot["trC"]),
                                              "W2 (trC2)": float(S[conf, 1].sum() / tot["trC2"]),
                                              "W0 (trC + |ubar|^2)": float((S[conf, 0] + S[conf, 2]).sum() / (tot["trC"] + tot["ub2"]))},
               ubar_share_of_W0_total_trace=float(tot["ub2"] / (tot["trC"] + tot["ub2"])))
    (R / f"runs/w11/m3_ctxweights_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
