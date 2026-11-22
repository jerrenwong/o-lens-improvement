"""ex2 S1 + S2 statistics and nulls (CPU, numpy), the S3 scales, and the S6 regressions.

Inputs: runs/ex2/gram.json (s1_gram.py), runs/ex2/s2_<fit>.pt (s2_func.py).
Writes runs/ex2/s1.json, s2.json, scales.json, s6.json.
Percentile of an observed statistic in a null = P(null < obs) + 0.5 P(null == obs)  (in %).
"""
import itertools
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
sys.path.insert(0, "/workspace/o-lens-improvement/src")
from ex2_common import BENCH, FITS, LAYERS, MAIN, OUT, RELEASED, load_matrix_dict, null_weights  # noqa: E402

gram = json.load(open(f"{OUT}/gram.json"))
PERMS = np.array(list(itertools.permutations(range(8))))  # all 40320


def pct(null, obs):
    null = np.asarray(null)
    return float(100 * ((null < obs).mean() + 0.5 * (null == obs).mean()))


def fro_stats(U, G):
    """U [k,8] weights; G 9x9 Gram over (E_0..E_7, Jbar). Returns dict of arrays [k]."""
    GE, g, GJ = G[:8, :8], G[:8, 8], G[8, 8]
    n2 = np.einsum("ke,ef,kf->k", U, GE, U)
    ip = U @ g
    l1 = np.abs(U).sum(1)
    c = U.sum(1)
    # deviations D_e = E_e - Jbar
    GD = GE - g[:, None] - g[None, :] + GJ
    dev2 = np.einsum("ke,ef,kf->k", U, GD, U)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(c_rel=c / l1, R2=ip ** 2 / (n2 * GJ), cos=ip / np.sqrt(n2 * GJ), beta=ip / GJ / l1,
                    share_cJ=c ** 2 * GJ / n2, share_dev=dev2 / n2)


# ---------------- S1 ----------------
s1 = {}
for fit, lay in gram["fits"].items():
    out = {}
    for L, d in lay.items():
        G = np.array(d["G"])[:9, :9]
        Gfull = np.array(d["G"])
        w = np.array(d["w"])
        P, N = np.clip(w, 0, None), np.clip(-w, 0, None)
        obs = {k: float(v[0]) for k, v in fro_stats(w[None], G).items()}
        obsP = {k: float(v[0]) for k, v in fro_stats(P[None], G).items()}
        obsN = {k: float(v[0]) for k, v in fro_stats(N[None], G).items()}
        nl = null_weights(w, seed=int(L))
        nulls = {"sign": nl["sign"], "perm": w[PERMS], "gauss": nl["gauss"]}
        r = dict(w=w.tolist(), c=float(w.sum()), L=obs, P=obsP, N=obsN,
                 cos_PN=float(P @ G[:8, :8] @ N / np.sqrt((P @ G[:8, :8] @ P) * (N @ G[:8, :8] @ N))) if N.any() else None,
                 identity_resid=d["identity_resid"], combine_resid=d["combine_resid"],
                 pooled16_vs_pooled=d["pooled16_vs_pooled"],
                 cos_E_J=(G[:8, 8] / np.sqrt(np.diag(G)[:8] * G[8, 8])).tolist(),
                 cos_Lfile_J=float(Gfull[9, 8] / np.sqrt(Gfull[9, 9] * Gfull[8, 8])),
                 pct={}, null_median={})
        for nn, U in nulls.items():
            st = fro_stats(U, G)
            r["pct"][nn] = {k: pct(st[k], obs[k]) for k in ("c_rel", "R2", "cos", "beta", "share_cJ")}
            r["pct"][nn]["abs_c_rel"] = pct(np.abs(st["c_rel"]), abs(obs["c_rel"]))
            r["null_median"][nn] = {k: float(np.nanmedian(st[k])) for k in ("c_rel", "R2", "cos", "share_cJ")}
            r["null_median"][nn]["abs_c_rel"] = float(np.median(np.abs(st["c_rel"])))
        # non-negative nulls for P and N: Dirichlet mixtures and permutations of P's / N's own magnitudes
        for part, vec in (("P", P), ("N", N)):
            if not vec.any():
                continue
            o = r[part]
            for nn, U in (("dir", nl["dir"]), ("perm", vec[PERMS])):
                st = fro_stats(U, G)
                r["pct"][f"{part}_{nn}"] = {k: pct(st[k], o[k]) for k in ("R2", "cos")}
                r["null_median"][f"{part}_{nn}"] = {k: float(np.nanmedian(st[k])) for k in ("R2", "cos")}
        out[L] = r
    # joint over layers: mean over layers of R2_L and |c_rel| vs independent per-layer null draws
    rng = np.random.default_rng(1)
    joint = {}
    for nn in ("sign", "perm", "gauss"):
        sims = {"R2": [], "abs_c_rel": []}
        for L, d in lay.items():
            G = np.array(d["G"])[:9, :9]
            w = np.array(d["w"])
            nl = null_weights(w, seed=int(L))
            U = {"sign": nl["sign"], "perm": w[PERMS], "gauss": nl["gauss"]}[nn]
            st = fro_stats(U[rng.integers(0, len(U), 20000)], G)
            sims["R2"].append(st["R2"])
            sims["abs_c_rel"].append(np.abs(st["c_rel"]))
        joint[nn] = {}
        for k in sims:
            nullmean = np.mean(sims[k], 0)
            obsmean = float(np.mean([out[L]["L"]["R2"] if k == "R2" else abs(out[L]["L"]["c_rel"]) for L in lay]))
            joint[nn][k] = dict(obs=obsmean, null_median=float(np.median(nullmean)), pct=pct(nullmean, obsmean))
    s1[fit] = dict(layers=out, joint=joint)
json.dump(s1, open(f"{OUT}/s1.json", "w"), indent=1)
print("[stats] s1 done", flush=True)

# ---------------- S2 + scales ----------------
s2, scales = {}, {}
for fit in FITS:
    try:
        d = torch.load(f"{OUT}/s2_{fit}.pt", weights_only=False)
    except FileNotFoundError:
        print("[stats] missing s2 for", fit)
        continue
    nf = d["n_fit"]
    out, sc = {}, {}
    jn = {}  # (set, null) -> list over layers of (null mean-frac array, observed)
    for L, lo in d["layers"].items():
        Gp = np.array(lo["G"])  # [P,10,10]: E0..E7, Jbar, Lfile
        w = np.array(gram["fits"][fit][L]["w"])
        Gfro = np.array(gram["fits"][fit][L]["G"])
        a = Gfro[9, 8] / Gfro[8, 8]  # Frobenius projection coefficient of L on Jbar
        fitrows = slice(0, nf)
        nJ = Gp[fitrows, 8, 8].sum()
        nL = Gp[fitrows, 9, 9].sum()
        nperp = (Gp[fitrows, 9, 9] - 2 * a * Gp[fitrows, 9, 8] + a * a * Gp[fitrows, 8, 8]).sum()
        sc[L] = dict(s_L=float(np.sqrt(nL / nJ)), a_fro=float(a), s_perp=float(np.sqrt(nperp / nJ)),
                     gamma_L=float(a / np.sqrt(nperp / nJ)), beta_equiv_pooled_share=None)

        def frac(U, rows):
            g = Gp[rows]  # [p,10,10]
            num = np.einsum("ke,pe->kp", U, g[:, :8, 8]) ** 2
            den = np.einsum("ke,pef,kf->kp", U, g[:, :8, :8], U) * g[:, 8, 8][None]
            return num / den  # [k,p]

        nl = null_weights(w, seed=int(L))
        P, N = np.clip(w, 0, None), np.clip(-w, 0, None)
        r = {}
        for setname, rows in (("fit", slice(0, nf)), ("ord", slice(nf, None))):
            fo = {nm: float(frac(v[None], rows).mean()) for nm, v in (("L", w), ("P", P), ("N", N)) if v.any()}
            rr = dict(obs=fo, pct={}, null_median={})
            for nn, U in (("sign", nl["sign"]), ("perm", nl["perm"]), ("gauss", nl["gauss"])):
                fm = frac(U, rows).mean(1)
                jn.setdefault((setname, nn), []).append((fm, fo["L"]))
                rr["pct"][nn] = pct(fm, fo["L"])
                rr["null_median"][nn] = float(np.median(fm))
            for part, vec in (("P", P), ("N", N)):
                if vec.any():
                    fm = frac(nl["dir"], rows).mean(1)
                    rr["pct"][f"{part}_dir"] = pct(fm, fo[part])
                    rr["null_median"][f"{part}_dir"] = float(np.median(fm))
            r[setname] = rr
        if "overlap" in lo:
            ov = lo["overlap"]
            dec = np.array(lo["dec_rows"])
            for setname, msk in (("fit", dec < nf), ("ord", dec >= nf)):
                o = {}
                for mode in ("f", "u"):
                    obsL = float(np.array(ov["L"][mode])[0, msk].mean())
                    o[mode] = dict(L=obsL, P=float(np.array(ov["P"][mode])[0, msk].mean()),
                                   N=float(np.array(ov["N"][mode])[0, msk].mean()) if N.any() else None)
                    possign = nl["sign"].sum(1) > 0  # orientation-fixed sign null: patterns with a positive net average
                    nmp = np.array(ov["sign"][mode])[possign][:, msk].mean(1)
                    o[mode]["pct_signpos"] = pct(nmp, obsL)
                    o[mode]["median_signpos"] = float(np.median(nmp))
                    for nn in ("sign", "perm", "dir"):
                        nm = np.array(ov[nn][mode])[:, msk].mean(1)
                        o[mode][f"pct_{nn}"] = pct(nm, obsL if nn != "dir" else o[mode]["N"] if N.any() else np.nan)
                        o[mode][f"median_{nn}"] = float(np.median(nm))
                r[setname]["overlap10"] = o
        out[L] = r
    rng = np.random.default_rng(2)
    joint = {}
    for (st, nn), lst in jn.items():
        sims = np.mean([fm[rng.integers(0, len(fm), 20000)] for fm, _ in lst], 0)
        obs = float(np.mean([o for _, o in lst]))
        joint[f"{st}_{nn}"] = dict(obs=obs, null_median=float(np.median(sims)), pct=pct(sims, obs))
    out["joint"] = joint
    s2[fit] = out
    scales[fit] = sc
json.dump(s2, open(f"{OUT}/s2.json", "w"), indent=1)
print("[stats] s2 done", flush=True)

# ---------------- S6 regressions + released scales on FIT activations ----------------
from olens.readout import Bench  # noqa: E402

s6 = {}
for m, lay in gram["released"].items():
    out = {}
    for L, d in lay.items():
        G = np.array(d["G"])
        names = d["names"]
        r = {}
        for j, nm in enumerate(names[1:], start=1):
            coef = G[0, j] / G[j, j]
            r[nm] = dict(coef=float(coef), R2=float(G[0, j] ** 2 / (G[0, 0] * G[j, j])),
                         cos=float(G[0, j] / np.sqrt(G[0, 0] * G[j, j])))
        if m in MAIN:  # joint regression on J-lens + main pooled
            jm = names.index(f"pool_{MAIN[m]}")
            idx = [1, jm]
            A = G[np.ix_(idx, idx)]
            b = G[idx, 0]
            beta = np.linalg.solve(A, b)
            r["joint_jl_pool"] = dict(coef=beta.tolist(), R2=float(beta @ b / G[0, 0]))
            r["cos_jl_pool"] = float(G[1, jm] / np.sqrt(G[1, 1] * G[jm, jm]))
        out[L] = r
    s6[m] = dict(layers=out)
# FIT-activation scales for the released dose-response (9B, 2B)
for m in ("9b", "2b"):
    bench = Bench.load(BENCH[m])
    fi = bench.index("fit")
    jpp = load_matrix_dict(RELEASED[m]["jpp"], LAYERS[m])
    jl = load_matrix_dict(RELEASED[m]["jl"], LAYERS[m])
    pool = load_matrix_dict(FITS[MAIN[m]]["pooled"], LAYERS[m])
    sc = {}
    for L in LAYERS[m]:
        h = bench.resid[L][fi].float()
        n = {k: float((h @ M[L].T).pow(2).sum()) for k, M in (("jpp", jpp), ("jl", jl), ("pool", pool))}
        Gr = np.array(gram["released"][m][str(L)]["G"])
        names = gram["released"][m][str(L)]["names"]
        jm = names.index(f"pool_{MAIN[m]}")
        a_jl, a_pool = Gr[0, 1] / Gr[1, 1], Gr[0, jm] / Gr[jm, jm]
        sc[str(L)] = dict(s_jl=float(np.sqrt(n["jpp"] / n["jl"])), s_pool=float(np.sqrt(n["jpp"] / n["pool"])),
                          a_jl=float(a_jl), a_pool=float(a_pool))
    scales[f"released_{m}"] = sc
json.dump(s6, open(f"{OUT}/s6.json", "w"), indent=1)
json.dump(scales, open(f"{OUT}/scales.json", "w"), indent=1)
print("[stats] s6 + scales done", flush=True)
