"""ex2 S2 (function space) + scales for S3 (GPU, lite model = final norm + LM head only).

Activations: (i) FIT readout activations = the bench's FIT rows (the residuals the weights were fitted on);
(ii) ORD = 2,000 ordinary-text activations: a seeded random sample of the e1 label-free prefixes' last-token
residuals (/root/e1/acts*.pt: WikiText detok / TinyStories / Wikipedia 10 languages; no benchmark text).
Per fit and layer, per activation h: the 10x10 Gram of [E_0 h..E_7 h, Jbar h, L h] (float64) -> saved; all
'fraction of ||Lh||^2 along Jbar h' statistics and their nulls follow from it (s2_stats.py).
Decoded lists (main fits only): top-10 of the readout of L h, P h, N h, Jbar h, filtered (bench.excluded_ids,
as the scorer) and unfiltered, and of null mixtures (256 sign patterns of |w|, 200 permutations of w,
200 Dirichlet non-negative mixtures), using the rank-exact linear trick of jpp_lens rank_objective
(unembed(x)*rms(x) is linear in x). Overlap = |top10(A) & top10(Jbar h)| / 10.
    python scripts/ex2/s2_func.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
sys.path.insert(0, "/workspace/o-lens-improvement/src")
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/perf1")
sys.path.insert(0, "/root/jpp_lens/src")
from ex2_common import BENCH, FITS, LAYERS, MAIN, MODEL, OUT, load_experts, load_matrix_dict, load_weights, null_weights  # noqa: E402
from lite_model import get_lite_model  # noqa: E402
from olens.readout import Bench  # noqa: E402
from workspace_lens.fitting.condense_experts.rank_objective import expert_logits  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(48)
dev = "cuda"
N_ORD, N_ORD_DEC = 2000, 500
ACTS = {"9b": "/root/e1/acts.pt", "2b": "/root/e1/acts_2b.pt"}
res = {}
t0 = time.time()
for m in ("2b", "9b"):
    bench = Bench.load(BENCH[m])
    fi = bench.index("fit")
    X = torch.load(ACTS[m], map_location="cpu", weights_only=False, mmap=True)["X"]
    n_all = X[LAYERS[m][0]].shape[0]
    ord_idx = np.sort(np.random.default_rng(0).choice(n_all, N_ORD, replace=False))
    H = {L: torch.cat([bench.resid[L][fi].float(), X[L][torch.as_tensor(ord_idx)].float()]) for L in LAYERS[m]}  # cpu
    n_fit = len(fi)
    model = get_lite_model(MODEL[m])
    excl = torch.zeros(model._lm_head.weight.shape[0], dtype=torch.bool, device=dev)
    excl[bench.excluded_ids.to(dev)] = True
    dec_rows = list(range(n_fit)) + list(range(n_fit, n_fit + N_ORD_DEC))  # FIT + first 500 ORD for decoding
    for fit in [f for f, c in FITS.items() if c["m"] == m]:
        if os.path.exists(f"{OUT}/s2_{fit}.pt"):
            print(f"[s2] skip {fit} (done)", flush=True)
            continue
        E, _, _, _ = load_experts(fit)
        Jb = load_matrix_dict(FITS[fit]["pooled"], LAYERS[m])
        Lf = load_matrix_dict(FITS[fit]["L"], LAYERS[m])
        W = load_weights(fit)
        out = {"n_fit": n_fit, "n_ord": N_ORD, "ord_idx": ord_idx.tolist(), "layers": {}}
        for L in LAYERS[m]:
            h = H[L]
            B = torch.cat([E[L].float(), Jb[L][None], Lf[L][None]])  # [10,F,N] cpu
            V = torch.einsum("pn,kfn->pkf", h, B)  # [P,10,F] cpu (matrix work on CPU: the GPU is shared)
            G = torch.einsum("pkf,plf->pkl", V.double(), V.double())
            lo = {"G": G.numpy().tolist()}
            if fit in MAIN.values():
                w = np.array(W[L])
                nl = null_weights(w)
                Wset = {"L": w[None], "P": np.clip(w, 0, None)[None], "N": np.clip(-w, 0, None)[None],
                        "sign": nl["sign"], "perm": nl["perm"][:200], "dir": nl["dir"][:200]}
                names = list(Wset)
                Wall = torch.tensor(np.concatenate([Wset[k] for k in names]), dtype=torch.float32, device=dev)
                Wall = torch.cat([torch.cat([Wall, torch.zeros(len(Wall), 1, device=dev)], 1),
                                  torch.tensor([[0.0] * 8 + [1.0]], device=dev)])  # last row = Jbar alone
                sizes = [len(Wset[k]) for k in names]
                ov = {k: {"f": [], "u": []} for k in names}
                for start in range(0, len(dec_rows), 16):
                    rows = dec_rows[start:start + 16]
                    lg = expert_logits(model.unembed, V[rows, :9].to(dev))  # [p,9,Vocab]
                    for mode in ("f", "u"):
                        tops = []
                        for ks in range(0, Wall.shape[0], 64):
                            comb = torch.einsum("ke,pev->kpv", Wall[ks:ks + 64], lg)
                            if mode == "f":
                                comb.masked_fill_(excl[None, None], float("-inf"))
                            tops.append(comb.topk(10, dim=-1).indices)
                            del comb
                        tops = torch.cat(tops)  # [K+1, p, 10]
                        ref = tops[-1]  # Jbar
                        hit = (tops[:-1, :, :, None] == ref[None, :, None, :]).any(-1).float().mean(-1)  # [K,p]
                        off = 0
                        for k, sz in zip(names, sizes, strict=True):
                            ov[k][mode].append(hit[off:off + sz].cpu())
                            off += sz
                    del lg
                lo["overlap"] = {k: {md: torch.cat(v[md], 1).numpy().round(3).tolist() for md in ("f", "u")}
                                 for k, v in ov.items()}  # [n_null, n_dec_rows]
                lo["dec_rows"] = dec_rows
            out["layers"][str(L)] = lo
            del V, B
            torch.cuda.empty_cache()
            print(f"[s2] {fit} L{L} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(out, f"{OUT}/s2_{fit}.pt")
        del E, Jb, Lf
    del model, H, X
    torch.cuda.empty_cache()
print("[s2] done", flush=True)
