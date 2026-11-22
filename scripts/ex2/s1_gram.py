"""ex2 S1/S6 step 1 (CPU): Frobenius Gram matrices.

Per fit and layer, basis B = [E_0..E_7, Jbar (pooled file, exact fp32), L (combined file), Jbar16 (the experts
file's fp16 pooled copy)]; Gram in float64. Every Frobenius statistic of any linear combination of these
(lens norms, projections, cosines, the null distributions of S1) follows from the Gram exactly.
Also the exact identity residuals:  ||Jbar - sum_e (n_e/n) E_e|| / ||Jbar||  and  ||L - sum_e w_e E_e|| / ||L||.
S6: per model, Gram of [released J++, released/stored J-lens, pooled lens of each fit of that model].
Writes runs/ex2/gram.json
    /root/jpp_lens/.venv/bin/python scripts/ex2/s1_gram.py
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
from ex2_common import FITS, LAYERS, OUT, RELEASED, load_experts, load_matrix_dict, load_weights  # noqa: E402

torch.set_num_threads(64)
res = {"fits": {}, "released": {}}
t0 = time.time()
for fit, cfg in FITS.items():
    m = cfg["m"]
    E, counts, pooled16, fallback = load_experts(fit)
    Lf = load_matrix_dict(cfg["L"], LAYERS[m])
    Jb = load_matrix_dict(cfg["pooled"], LAYERS[m])
    W = load_weights(fit)
    out = {}
    for L in LAYERS[m]:
        Ee = E[L].float()
        n = counts[L]
        B = torch.cat([Ee.reshape(8, -1), Jb[L].reshape(1, -1), Lf[L].reshape(1, -1),
                       pooled16[L].float().reshape(1, -1)]).double()
        G = (B @ B.T)
        mean_e = torch.einsum("e,ef->f", (n / n.sum()), Ee.reshape(8, -1).double())
        id_res = float((Jb[L].reshape(-1).double() - mean_e).norm() / Jb[L].double().norm())
        id_res16 = float((pooled16[L].double().reshape(-1) - Jb[L].double().reshape(-1)).norm() / Jb[L].double().norm())
        w = torch.tensor(W[L], dtype=torch.float64)
        comb = torch.einsum("e,ef->f", w, Ee.reshape(8, -1).double())
        comb_res = float((Lf[L].reshape(-1).double() - comb).norm() / Lf[L].double().norm())
        out[L] = dict(G=G.tolist(), counts=n.tolist(), fallback=fallback[L].tolist(), w=W[L],
                      identity_resid=id_res, pooled16_vs_pooled=id_res16, combine_resid=comb_res)
        print(f"{fit} L{L}: identity {id_res:.2e}  fp16-pooled {id_res16:.2e}  combine {comb_res:.2e}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del B, Ee
    res["fits"][fit] = out
    del E, Lf, Jb, pooled16

for m, rel in RELEASED.items():
    lay = LAYERS[m]
    jpp = load_matrix_dict(rel["jpp"], lay)
    jl = load_matrix_dict(rel["jl"], lay)
    fits = [f for f, c in FITS.items() if c["m"] == m and f != "9b_w2"]
    pools = {f: load_matrix_dict(FITS[f]["pooled"], lay) for f in fits}
    out = {}
    for L in lay:
        B = torch.cat([jpp[L].reshape(1, -1), jl[L].reshape(1, -1)] + [pools[f][L].reshape(1, -1) for f in fits]).double()
        out[L] = dict(names=["jpp", "jl"] + [f"pool_{f}" for f in fits], G=(B @ B.T).tolist())
        print(f"released {m} L{L} done ({time.time() - t0:.0f}s)", flush=True)
    res["released"][m] = out
    del jpp, jl, pools

json.dump(res, open(f"{OUT}/gram.json", "w"))
print("wrote", f"{OUT}/gram.json", flush=True)
