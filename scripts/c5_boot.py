"""Paired bootstrap for auto/PREREG_v4.md (third fresh set). Metric and resampling as scripts/c3_boot.py: per (item,
intermediate) best rank over layers; per item share of intermediates with rank <= k; mean per task; macro over tasks;
items resampled within model x task, 5000 draws, same draws for all lenses of a model. 'both' = mean of the two models."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement"); NB = 5000
KEYS = ["J", "L", "C", "G", "Cs"]
NAME = dict(J="jpp", L="L", C="C3", G="G", Cs="C3shuf")
def load(v, t, kind, k):
    df = pd.concat([pd.read_csv(R / "auto/runs" / f"V{v}_{NAME[n]}_{t}" / f"readout_{kind}.csv").assign(lens=n) for n in KEYS])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[KEYS].isna().any().any()
    return {e: item.loc[e][KEYS].to_numpy() for e in sorted(item.index.get_level_values(0).unique())}
def boot(per, seed):
    rng = np.random.default_rng(seed); ev = sorted(per)
    point = np.mean([per[e].mean(0) for e in ev], 0); b = np.empty((NB, len(KEYS)))
    for i in range(NB):
        b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in ev], 0)
    return point, b
def line(name, p, b, x, y):
    i, j = KEYS.index(x), KEYS.index(y); d = b[:, i] - b[:, j]
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return f"  {name:14s} {x:2s}-{y:2s} {100*(p[i]-p[j]):+5.1f} [{100*np.percentile(d,2.5):+5.1f},{100*np.percentile(d,97.5):+5.1f}] p~{pv:.4f}  ({100*p[j]:.1f} -> {100*p[i]:.1f})"
PAIRS = [("L", "J"), ("C", "J"), ("G", "J"), ("L", "C"), ("Cs", "J")]
for kind, k in (("filtered", 10), ("filtered", 1), ("unfiltered", 10)):
    print(f"\n==== {kind} pass@{k}")
    res = {(v, t): boot(load(v, t, kind, k), seed=v * 10 + (0 if t == "9b" else 1)) for v in (4, 3, 2) for t in ("9b", "27b")}
    if kind == "filtered" and k == 10:
        per9, per27 = load(4, "9b", kind, k), load(4, "27b", kind, k)
        for t, per in (("9b", per9), ("27b", per27)):
            n = sum(len(x) for x in per.values()); bt = sum((x[:, 1] > x[:, 0]).sum() for x in per.values()); ws = sum((x[:, 1] < x[:, 0]).sum() for x in per.values())
            print(f"  [v4 {t}: {n} prompts; L better than J++ on {bt}, worse on {ws}; per task L-J: " + " ".join(f"{e.split('-')[-1][:6]} {100*(x[:,1].mean()-x[:,0].mean()):+.1f}" for e, x in per.items()) + "]")
    for x, y in PAIRS:
        for t in ("9b", "27b"):
            print(line(f"v4 {t}", *res[(4, t)], x, y))
        p = (res[(4, "9b")][0] + res[(4, "27b")][0]) / 2; b = (res[(4, "9b")][1] + res[(4, "27b")][1]) / 2
        print(line("v4 BOTH", p, b, x, y))
        for t in ("9b", "27b"):
            p = sum(res[(v, t)][0] for v in (2, 3, 4)) / 3; b = sum(res[(v, t)][1] for v in (2, 3, 4)) / 3
            print(line(f"v2+v3+v4 {t}", p, b, x, y))
        p = sum(r[0] for r in res.values()) / 6; b = sum(r[1] for r in res.values()) / 6
        print(line("v2+v3+v4 BOTH", p, b, x, y)); print()
