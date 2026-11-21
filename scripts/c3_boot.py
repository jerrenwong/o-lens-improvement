"""Paired bootstrap for PREREG_v3 amendment 5 / PREREG_v2 amendment 3 (clean chat-format ask-map C).
Metric as scripts/r2_boot.py: per (item, intermediate) best rank over layers; per item share of intermediates with
rank <= k; mean per task; macro over tasks. Items resampled within task, 5000 draws, same draws for all lenses of a
(set, model). 'both models' = mean of the two models' macro scores; 'v2+v3' = mean over the two sets."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement"); NB = 5000
L = {"9b": dict(J="jpp_9b", A="e2_A_9b", As="e2_C_shuf_9b", C="C3_9b", Cs="C3shuf_9b"),
     "27b": dict(J="jpp_27b", A="m27_A_27b", As="m27_C_shuf_27b", C="C3_27b", Cs="C3shuf_27b")}
KEYS = ["J", "A", "As", "C", "Cs"]
def load(v, t, kind, k):
    df = pd.concat([pd.read_csv(R / "auto/runs" / f"V{v}_{L[t][n]}" / f"readout_{kind}.csv").assign(lens=n) for n in KEYS])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[KEYS].isna().any().any()
    return {e: item.loc[e][KEYS].to_numpy() for e in sorted(item.index.get_level_values(0).unique())}
def boot(per, seed):
    rng = np.random.default_rng(seed); ev = sorted(per)
    point = np.mean([per[e].mean(0) for e in ev], 0)
    b = np.empty((NB, len(KEYS)))
    for i in range(NB):
        b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in ev], 0)
    return point, b
def line(name, p, b, x, y):
    i, j = KEYS.index(x), KEYS.index(y); d = b[:, i] - b[:, j]
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return f"  {name:28s} {x}-{y:2s} {100*(p[i]-p[j]):+5.1f} [{100*np.percentile(d,2.5):+5.1f},{100*np.percentile(d,97.5):+5.1f}] p~{pv:.3f}  ({100*p[j]:.1f} -> {100*p[i]:.1f})"
PAIRS = [("C", "J"), ("C", "A"), ("C", "Cs"), ("A", "J"), ("Cs", "J")]
for kind, k in (("filtered", 10), ("filtered", 1), ("unfiltered", 10)):
    print(f"\n==== {kind} pass@{k}")
    res = {}
    for v in (3, 2):
        for t in ("9b", "27b"):
            per = load(v, t, kind, k); res[(v, t)] = boot(per, seed=v * 10 + (0 if t == "9b" else 1))
            if kind == "filtered" and k == 10:
                n = sum(len(x) for x in per.values())
                bw = sum(((x[:, 3] > x[:, 0]).sum(), 0)[0] for x in per.values()), sum((x[:, 3] < x[:, 0]).sum() for x in per.values())
                print(f"  [V{v} {t}: {n} prompts; C better than J++ on {bw[0]}, worse on {bw[1]}]")
    for x, y in PAIRS:
        for v in (3, 2):
            for t in ("9b", "27b"):
                print(line(f"v{v} {t}", *res[(v, t)], x, y))
            p = (res[(v, "9b")][0] + res[(v, "27b")][0]) / 2; b = (res[(v, "9b")][1] + res[(v, "27b")][1]) / 2
            print(line(f"v{v} BOTH MODELS", p, b, x, y))
        for t in ("9b", "27b"):
            p = (res[(2, t)][0] + res[(3, t)][0]) / 2; b = (res[(2, t)][1] + res[(3, t)][1]) / 2
            print(line(f"v2+v3 {t}", p, b, x, y))
        p = sum(r[0] for r in res.values()) / 4; b = sum(r[1] for r in res.values()) / 4
        print(line("v2+v3 BOTH MODELS", p, b, x, y)); print()
