"""ex2: compile runs/ex2/report.md and the figure data runs/ex2/fig_dose.json from the ex2 outputs."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
from ex2_common import FITS, LAYERS, MAIN, OUT, R, null_weights  # noqa: E402
from s12_stats_lib import fro_stats, pct  # noqa: E402

rep = []
P = rep.append
s1 = json.load(open(f"{OUT}/s1.json"))
s2 = json.load(open(f"{OUT}/s2.json")) if os.path.exists(f"{OUT}/s2.json") else {}
s6 = json.load(open(f"{OUT}/s6.json"))
scales = json.load(open(f"{OUT}/scales.json"))
gram = json.load(open(f"{OUT}/gram.json"))["fits"]
BOOT = {}
for m in ("9b", "2b"):
    if os.path.exists(f"{OUT}/boot_{m}.json"):
        b = json.load(open(f"{OUT}/boot_{m}.json"))
        b["draws"] = {k: np.load(f"{OUT}/boots_{m}_{k}.npy") for k in ("p10", "p1", "p10u")}
        BOOT[m] = b
FIG = {}


def has(m, n):
    return m in BOOT and n in BOOT[m]["names"]


def pt(m, n, k="p10"):
    return 100 * BOOT[m]["metrics"][k]["point"][n]


def dlt(m, a, b, k="p10"):
    B = BOOT[m]
    i, j = B["names"].index(a), B["names"].index(b)
    d = 100 * (B["draws"][k][:, i] - B["draws"][k][:, j])
    return 100 * (B["metrics"][k]["point"][a] - B["metrics"][k]["point"][b]), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fd(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def share(m, x, L, pool, k="p10"):
    B = BOOT[m]
    i, j, q = (B["names"].index(n) for n in (x, L, pool))
    D = B["draws"][k]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (D[:, i] - D[:, q]) / (D[:, j] - D[:, q])
    p = B["metrics"][k]["point"]
    return (100 * (p[x] - p[pool]) / (p[L] - p[pool]), 100 * np.nanpercentile(r, 2.5), 100 * np.nanpercentile(r, 97.5),
            100 * float(np.mean(D[:, j] - D[:, q] > 0)))


P("# ex2: is J++ 'positive experts minus the average lens'?  (closed-ended study, DEV-only scores)\n")
P("All scores: auto/evaluate.py --lite on DEV (9B n=156 items, 2B n=109), macro pass@10 filtered unless noted, in points. "
  "Intervals: paired bootstrap 95% (5000 draws, items within task, same draws for all lenses; scripts/ex2/ex2_boot.py = r2_boot algorithm). "
  "Null percentile = % of null values below the observed one (ties half). Everything chosen (alphas, refits, scales) used FIT only.\n")
P("Fits: 9b_64a / 9b_64b = experts from two disjoint 48-prompt halves (shards 0-1 / 2-3), 9b_full = all 96 prompts; one shared "
  "router, so expert e is the same cluster in all 9B fits; 9b_w2 = second fit-weights run on the 64a experts (bit-identical weights). "
  "2b_A raw/skip16 (recipe), 2b_B natural/skip16, 2b_C natural/skip1, 2b_D raw/skip1; each 128 prompts with its own router "
  "(expert indices not comparable across 2B fits). Weights fitted on 75 (9B) / 52 (2B) FIT items.\n")

# ---------------- S0 identity ----------------
P("## S0 identity checks\n")
P("| fit | max_l ||Jbar - sum n_e/n E_e|| / ||Jbar|| | max_l ||L - sum w_e E_e|| / ||L|| | w2 == 64a weights |")
P("|---|---|---|---|")
for f, d in s1.items():
    ir = max(r["identity_resid"] for r in d["layers"].values())
    cr = max(r["combine_resid"] for r in d["layers"].values())
    same = "" if f != "9b_w2" else str(all(np.allclose(gram["9b_w2"][L]["w"], gram["9b_64a"][L]["w"], atol=0, rtol=0) for L in gram["9b_w2"]))
    P(f"| {f} | {ir:.1e} | {cr:.1e} | {same} |")
P("\nJbar is the n_e-weighted mean of the experts up to the experts' fp16 storage (1e-4); so L = c Jbar + sum_e w_e D_e holds to 1e-4.\n")

# ---------------- S1 ----------------
P("## S1 algebra (Frobenius), with nulls\n")
P("c_rel = sum w / sum |w|; R2 = fraction of ||X||_F^2 explained by Jbar (= cos^2); beta = LS coefficient of X on Jbar with ||w||_1 = 1. "
  "Nulls: sign = all 256 sign patterns of |w| (exact); perm = all 40,320 permutations of w; gauss = 2,000 random directions; "
  "N/P dir = 2,000 Dirichlet non-negative mixtures. pct < 50 = closer to cancelling the average than the null's median.\n")
for m in ("9b", "2b"):
    for f in [f for f in s1 if FITS[f]["m"] == m and f != "9b_w2"]:
        d = s1[f]
        P(f"\n**{f}**\n")
        P("| layer | w (unit L2) | c_rel | R2_L | beta_L | share c^2||Jbar||^2/||L||^2 | pct sign R2 / abs c_rel | pct perm R2 | pct gauss R2 | R2_P (pct dir) | R2_N (pct dir) | med R2 dir |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for L, r in d["layers"].items():
            w = " ".join(f"{x:+.2f}" for x in r["w"])
            pn = r["pct"].get("N_dir", {}).get("R2", float("nan"))
            P(f"| {L} | {w} | {r['L']['c_rel']:+.2f} | {r['L']['R2']:.2f} | {r['L']['beta']:+.2f} | {r['L']['share_cJ']:.2f} | "
              f"{r['pct']['sign']['R2']:.0f} / {r['pct']['sign']['abs_c_rel']:.0f} | {r['pct']['perm']['R2']:.0f} | "
              f"{r['pct']['gauss']['R2']:.0f} | {r['P']['R2']:.2f} ({r['pct']['P_dir']['R2']:.0f}) | "
              f"{r['N']['R2']:.2f} ({pn:.0f}) | {r['null_median']['N_dir']['R2']:.2f} |")
        j = d["joint"]
        P(f"| mean | | {np.mean([r['L']['c_rel'] for r in d['layers'].values()]):+.2f} | {j['sign']['R2']['obs']:.2f} | | | "
          f"{j['sign']['R2']['pct']:.0f} / {j['sign']['abs_c_rel']['pct']:.0f} | {j['perm']['R2']['pct']:.0f} | {j['gauss']['R2']['pct']:.0f} | | | |")
P("\n(joint row: mean over the 7 layers vs 20,000 draws of independent per-layer nulls)\n")

# ---------------- S2 ----------------
if s2:
    P("## S2 function space (activations)\n")
    P("frac = mean over activations of the fraction of ||X h||^2 along Jbar h. FIT = FIT readout activations; ORD = 2,000 "
      "ordinary-text activations (random e1 label-free prefixes, last token). pct vs sign / perm nulls (same weights as S1). "
      "overlap = |top10(L h) & top10(Jbar h)|/10, filtered (scorer's filter), main fits only; null = the 128 sign patterns of |w| with a positive net sum (orientation fixed), median and percentile.\n")
    for m in ("9b", "2b"):
        for f in [f for f in s2 if FITS[f]["m"] == m and f != "9b_w2"]:
            P(f"\n**{f}**\n")
            P("| layer | FIT frac L (pct sign/perm) | FIT frac P / N (N pct dir) | ORD frac L (pct sign/perm) | ORD frac P / N (N pct dir) | overlap10 FIT L (null med; pct) | overlap10 ORD L (null med; pct) | overlap10 FIT N / P |")
            P("|---|---|---|---|---|---|---|---|")
            for L, r in s2[f].items():
                if L == "joint":
                    continue
                cells = []
                for st in ("fit", "ord"):
                    o = r[st]
                    cells.append(f"{o['obs']['L']:.2f} ({o['pct']['sign']:.0f}/{o['pct']['perm']:.0f})")
                    cells.append(f"{o['obs']['P']:.2f} / {o['obs'].get('N', float('nan')):.2f} ({o['pct'].get('N_dir', float('nan')):.0f})")
                ov = []
                for st in ("fit", "ord"):
                    if "overlap10" in r[st]:
                        x = r[st]["overlap10"]["f"]
                        ov.append(f"{x['L']:.2f} ({x['median_signpos']:.2f}; {x['pct_signpos']:.0f})")
                    else:
                        ov.append("")
                x = r["fit"].get("overlap10", {}).get("f")
                ov.append(f"{x['N']:.2f} / {x['P']:.2f}" if x and x["N"] is not None else "")
                P(f"| {L} | " + " | ".join(cells + ov) + " |")
            j = s2[f]["joint"]
            P(f"| mean over layers | {j['fit_sign']['obs']:.2f} (joint pct {j['fit_sign']['pct']:.0f}/{j['fit_perm']['pct']:.0f}) | | "
              f"{j['ord_sign']['obs']:.2f} (joint pct {j['ord_sign']['pct']:.0f}/{j['ord_perm']['pct']:.0f}) | | | | |")


# ---------------- S3 ----------------
BETAS = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8]
GAMMAS = [0, 0.25, 0.5, 1, 2, 4, 8]


def nm_beta(f, b):
    return f"{f}_L" if b == 0 else f"{f}_b{b}"


P("\n## S3 dose-response: add the average back (all layers)\n")
P("L_beta = L + beta s_L Jbar (s_L = RMS||Lh||/RMS||Jbar h|| on FIT activations, per layer). Perp family: L_perp = L - a Jbar "
  "(Frobenius projection removed) + gamma s_perp Jbar; L itself sits at gamma_L (listed). Delta vs L with 95% CI.\n")
for m in ("9b", "2b"):
    for f in [f for f, c in FITS.items() if c["m"] == m and f != "9b_w2"]:
        if not has(m, f"{f}_L"):
            continue
        P(f"\n**{f}**  (gamma_L per layer: " + ", ".join(f"{scales[f][L]['gamma_L']:.2f}" for L in scales[f]) +
          "; s_L per layer: " + ", ".join(f"{scales[f][L]['s_L']:.2f}" for L in scales[f]) + ")\n")
        P("| lens | p10 | d p10 vs L | p1 | d p1 vs L | p10 unf | d p10u vs L |")
        P("|---|---|---|---|---|---|---|")
        fig = {"beta": [], "gamma": []}
        rows = [(f"beta={b}", nm_beta(f, b)) for b in BETAS] + [(f"perp gamma={g}", f"{f}_perp_g{g}") for g in GAMMAS] + \
            [("pooled (beta=inf)", f"{f}_pool")]
        for lab, n in rows:
            if not has(m, n):
                continue
            cells = [lab]
            for k in ("p10", "p1", "p10u"):
                cells += [f"{pt(m, n, k):.1f}", fd(dlt(m, n, f"{f}_L", k)) if n != f"{f}_L" else ""]
            P("| " + " | ".join(cells) + " |")
            entry = {k: pt(m, n, k) for k in ("p10", "p1", "p10u")}
            entry.update({f"d_{k}_ci": list(dlt(m, n, f"{f}_L", k)) for k in ("p10", "p1", "p10u")})
            if lab.startswith("beta"):
                fig["beta"].append(dict(beta=float(lab.split("=")[1]), **entry))
            elif lab.startswith("perp"):
                fig["gamma"].append(dict(gamma=float(lab.split("=")[1]), **entry))
            else:
                fig["pooled"] = entry
        FIG[f] = fig
        if has(m, f"{f}_b8"):
            pts = [pt(m, nm_beta(f, b)) for b in BETAS if b >= 0 and has(m, nm_beta(f, b))]
            mono = all(x >= y - 1e-9 for x, y in zip(pts, pts[1:]))
            P(f"\nnon-increasing over beta>=0: {mono}; L - pooled = {fd(dlt(m, f'{f}_L', f'{f}_pool'))}")
        # per task for main fit
        if f == MAIN[m]:
            P("\nper task p10 (main fit):\n")
            tasks = sorted(BOOT[m]["metrics"]["p10"]["n_items"])
            P("| lens | " + " | ".join(f"{t} (n={BOOT[m]['metrics']['p10']['n_items'][t]})" for t in tasks) + " |")
            P("|---|" + "---|" * len(tasks))
            for lab, n in rows:
                if has(m, n):
                    pt_ = BOOT[m]["metrics"]["p10"]["per_task"][n]
                    P(f"| {lab} | " + " | ".join(f"{100 * pt_[t]:.1f}" for t in tasks) + " |")

# ---------------- S4 ----------------
P("\n## S4 constructive: one-scalar contrasts without label-fitted signed weights\n")
P("share = (X - pooled)/(L - pooled) of the DEV gain, bootstrap CI; P(L>pooled) = share of draws with a positive denominator. "
  "alpha chosen on FIT (grid 0..2 in units of the Frobenius projection coefficient; 1 = Jbar component removed exactly).\n")
for m in ("9b", "2b"):
    for f in [f for f, c in FITS.items() if c["m"] == m and f != "9b_w2"]:
        if not (has(m, f"{f}_L") and os.path.exists(f"{OUT}/s4_{f}.json")):
            continue
        s4 = json.load(open(f"{OUT}/s4_{f}.json"))
        fs = s4["fit_scores"]
        P(f"\n**{f}** (FIT p10: L {100 * fs['L']:.1f}, pooled {100 * fs['pooled']:.1f}, sum0 {100 * fs['sum0']:.1f}; "
          f"single-expert pick e*/layer: {[s4['layers'][L]['estar'] for L in s4['layers']]}; content experts/layer: "
          f"{[s4['layers'][L]['content'] for L in s4['layers']]})\n")
        P("| lens | alpha (FIT) | FIT p10 | DEV p10 | d vs L | d vs pooled | share of L-pooled gain | p1 | p10 unf |")
        P("|---|---|---|---|---|---|---|---|---|")
        rows = [("(i) pooled", f"{f}_pool", None), ("(ii) best single expert - a Jbar", f"{f}_s4single", "single"),
                ("(iii) content mean - a Jbar", f"{f}_s4content", "content"), ("P alone (ref)", f"{f}_s4P0", None),
                ("(iv) P - a Jbar", f"{f}_s4P", "P"), ("(v) refit, sum w = 0", f"{f}_s4sum0", None),
                ("(vi) L", f"{f}_L", None), ("(vii) released J++", f"{m}rel_jpp", None), ("(vii) J-lens", f"{m}rel_jl", None)]
        for lab, n, fam in rows:
            if not has(m, n):
                continue
            al = s4.get(f"alpha_{fam}", "") if fam else ""
            fsc = fs.get(fam, {}).get(str(al), None) if fam else None
            if fam is None and n.endswith("sum0"):
                fsc = fs["sum0"]
            sh = share(m, n, f"{f}_L", f"{f}_pool")
            P(f"| {lab} | {al} | {'' if fsc is None else f'{100 * fsc:.1f}'} | {pt(m, n):.1f} | "
              f"{fd(dlt(m, n, f'{f}_L')) if n != f'{f}_L' else ''} | {fd(dlt(m, n, f'{f}_pool')) if n != f'{f}_pool' else ''} | "
              f"{sh[0]:.0f}% [{sh[1]:.0f}, {sh[2]:.0f}] (P(L>pool) {sh[3]:.0f}%) | {pt(m, n, 'p1'):.1f} | {pt(m, n, 'p10u'):.1f} |")

# ---------------- S5 ----------------
P("\n## S5 stability\n")
for m in ("9b", "2b"):
    f = MAIN[m]
    if not os.path.exists(f"{OUT}/s5_{f}.json"):
        continue
    s5 = json.load(open(f"{OUT}/s5_{f}.json"))
    for tag in ("_p1", "_p2", "_p3"):  # layer-split parallel runs (same seeded draws)
        if os.path.exists(f"{OUT}/s5_{f}{tag}.json"):
            sp = json.load(open(f"{OUT}/s5_{f}{tag}.json"))
            assert sp["draws"] == s5["draws"]
            for k in ("weights", "train"):
                s5[k].update({L: v for L, v in sp[k].items() if len(v) == len(sp["draws"])})
    s5["weights"] = {L: s5["weights"][L] for L in sorted(s5["weights"], key=int) if len(s5["weights"][L]) == len(s5["draws"])}
    nb = len(s5["draws"]) - 1
    P(f"\n**{f}: {nb} bootstrap resamples of the FIT items (within task), weights refit with the authors' search**\n")
    P("| layer | reproduction max abs(w - stored) | sign agreement with fit (mean over experts) | same negative set | P(w_e<0) per expert | c_rel fit / median [95%] | pct sign R2: fit / median [95%] | frac draws pct<50 |")
    P("|---|---|---|---|---|---|---|---|")
    joint_draw = []
    for L in s5["weights"]:
        G = np.array(gram[f][L]["G"])[:9, :9]
        W = np.array(s5["weights"][L])
        w0 = np.array(s5["stored"][L])
        rep_err = float(np.abs(W[0] - w0).max())
        Wb = W[1:]
        neg = Wb < 0
        agree = float((np.sign(Wb) == np.sign(w0)[None]).mean())
        sameset = float((neg == (w0 < 0)[None]).all(1).mean())
        st = fro_stats(Wb, G)
        cr = st["c_rel"]
        pcts = []
        for w in Wb:
            ns = fro_stats(null_weights(w)["sign"], G)
            pcts.append(pct(ns["R2"], fro_stats(w[None], G)["R2"][0]))
        pcts = np.array(pcts)
        joint_draw.append(pcts)
        c0 = s1[f]["layers"][L]
        P(f"| {L} | {rep_err:.1e} | {agree:.2f} | {sameset:.2f} | " + " ".join(f"{x:.2f}" for x in neg.mean(0)) +
          f" | {c0['L']['c_rel']:+.2f} / {np.median(cr):+.2f} [{np.percentile(cr, 2.5):+.2f}, {np.percentile(cr, 97.5):+.2f}] | "
          f"{c0['pct']['sign']['R2']:.0f} / {np.median(pcts):.0f} [{np.percentile(pcts, 2.5):.0f}, {np.percentile(pcts, 97.5):.0f}] | "
          f"{(pcts < 50).mean():.2f} |")
    jd = np.mean(joint_draw, 0)
    P(f"\nmean over layers of the per-layer sign-null percentile: fit {np.mean([s1[f]['layers'][L]['pct']['sign']['R2'] for L in s5['weights']]):.0f}, "
      f"bootstrap median {np.median(jd):.0f} [{np.percentile(jd, 2.5):.0f}, {np.percentile(jd, 97.5):.0f}]\n")
P("\n**Across 9B fits (shared router: expert e = same cluster)**\n")
P("| layer | negative experts 64a | 64b | full | c_rel 64a / 64b / full | R2_L 64a / 64b / full |")
P("|---|---|---|---|---|---|")
for L in s1["9b_64a"]["layers"]:
    negs = [[e for e, x in enumerate(s1[f]["layers"][L]["w"]) if x < 0] for f in ("9b_64a", "9b_64b", "9b_full")]
    P(f"| {L} | {negs[0]} | {negs[1]} | {negs[2]} | " + " / ".join(f"{s1[f]['layers'][L]['L']['c_rel']:+.2f}" for f in ("9b_64a", "9b_64b", "9b_full")) +
      " | " + " / ".join(f"{s1[f]['layers'][L]['L']['R2']:.2f}" for f in ("9b_64a", "9b_64b", "9b_full")) + " |")
P("\nKey S3/S4 numbers across fits (p10 points, d vs L [CI]):\n")
P("| fit | L | pooled | L - pooled | beta=+1 | beta=-0.5 | perp gamma=0 | P - aJbar | sum0 | single | content |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for f in [f for f in FITS if f != "9b_w2"]:
    m = FITS[f]["m"]
    if not has(m, f"{f}_L"):
        continue
    cells = [f, f"{pt(m, f'{f}_L'):.1f}", f"{pt(m, f'{f}_pool'):.1f}" if has(m, f"{f}_pool") else "",
             fd(dlt(m, f"{f}_L", f"{f}_pool")) if has(m, f"{f}_pool") else ""]
    for n in (f"{f}_b1", f"{f}_b-0.5", f"{f}_perp_g0", f"{f}_s4P", f"{f}_s4sum0", f"{f}_s4single", f"{f}_s4content"):
        cells.append(fd(dlt(m, n, f"{f}_L")) if has(m, n) else "")
    P("| " + " | ".join(cells) + " |")

# ---------------- S6 ----------------
P("\n## S6 released J++ (final matrices only: experts and weights are NOT available for them)\n")
P("Regression of the released J++ matrix on (a) the released/stored J-lens (9B: our stored J-lens runs/v6_9b, 48 prompts, no LRP; "
  "2B: neuronpedia J-lens, 283 prompts; 27B: released J-lens, 1000 prompts) and (b) our pooled LRP lens (9B 64a, 2B A); Frobenius.\n")
P("| model | layer | coef on J-lens | R2 J-lens | coef on pooled | R2 pooled | joint R2 (J-lens + pooled) | cos(J-lens, pooled) |")
P("|---|---|---|---|---|---|---|---|")
for m in ("2b", "9b", "27b"):
    for L, r in s6[m]["layers"].items():
        pk = f"pool_{MAIN[m]}" if m in MAIN else None
        P(f"| {m} | {L} | {r['jl']['coef']:+.3f} | {r['jl']['R2']:.3f} | " +
          (f"{r[pk]['coef']:+.3f} | {r[pk]['R2']:.3f} | {r['joint_jl_pool']['R2']:.3f} | {r['cos_jl_pool']:+.3f} |" if pk else "n/a | n/a | n/a | n/a |"))
for m in ("9b", "2b"):
    if not has(m, f"{m}rel_jpp"):
        continue
    P(f"\n**{m} released J++: add back the J-lens / the pooled LRP lens (s from FIT activations)**\n")
    P("| beta | +J-lens p10 | d vs J++ | +pooled p10 | d vs J++ |")
    P("|---|---|---|---|---|")
    fig = {"jl": [], "pool": []}
    for b in [-1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8]:
        a, c = f"{m}rel_jl_b{b}", f"{m}rel_pool_b{b}"
        cells = [str(b)]
        for n, key in ((a, "jl"), (c, "pool")):
            if has(m, n):
                cells += [f"{pt(m, n):.1f}", fd(dlt(m, n, f"{m}rel_jpp"))]
                fig[key].append(dict(beta=b, p10=pt(m, n), d_ci=list(dlt(m, n, f"{m}rel_jpp"))))
            else:
                cells += ["", ""]
        P("| " + " | ".join(cells) + " |")
    for n, lab in ((f"{m}rel_jl_perp0", "J++ minus its J-lens projection"), (f"{m}rel_pool_perp0", "J++ minus its pooled projection"),
                   (f"{m}rel_jl", "J-lens alone"), (f"{MAIN[m]}_pool", "pooled alone")):
        if has(m, n):
            P(f"| {lab} | {pt(m, n):.1f} | {fd(dlt(m, n, f'{m}rel_jpp'))} | | |")
    P(f"\nreleased J++ p10 {pt(m, f'{m}rel_jpp'):.1f}, p1 {pt(m, f'{m}rel_jpp', 'p1'):.1f}, unf {pt(m, f'{m}rel_jpp', 'p10u'):.1f}")
    FIG[f"released_{m}"] = fig

# ---------------- S7 ----------------
f = "9b_64a"
if has("9b", f"{f}_drop_L4_e0"):
    S = json.load(open(f"{R}/runs/ex1/9b/summary_9b.json"))
    P("\n## S7 leave-one-(layer, expert)-out on DEV (9b_64a): d p10 vs L\n")
    P("| layer | e | w | d p10 [CI] | d p1 [CI] | cluster (ex1: share of WikiText positions; dominant kind; frequent tokens) |")
    P("|---|---|---|---|---|---|")
    for L in LAYERS["9b"]:
        for e in range(8):
            n = f"{f}_drop_L{L}_e{e}"
            if not has("9b", n):
                continue
            ex = S["layers"][str(L)]["experts"][e]
            kd = max(ex["kinds"], key=ex["kinds"].get)
            toks = " ".join(t[0].replace(" ", "␣").replace("|", "/") for t in ex["freq"][:6])
            P(f"| {L} | {e} | {gram[f][str(L)]['w'][e]:+.2f} | {fd(dlt('9b', n, f'{f}_L'))} | {fd(dlt('9b', n, f'{f}_L', 'p1'))} | "
              f"{ex['frac']:.2f}; {kd} {ex['kinds'][kd]:.2f}; {toks} |")

json.dump(FIG, open(f"{OUT}/fig_dose.json", "w"), indent=1)
open(f"{OUT}/report.md", "w").write("\n".join(rep) + "\n")
print("wrote report.md")
