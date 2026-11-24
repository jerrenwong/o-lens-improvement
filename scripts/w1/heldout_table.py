"""W1 held-out table (PREREG_v5 Amendment 2): five rows x f1/f2/f3 x test/v4 per model, plus paired bootstrap of the
registered lens-vs-lens differences within each filter level. Usage: python scripts/w1/heldout_table.py [tags...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement/auto/runs")
TAGS = sys.argv[1:] or ["2b", "9b", "27b", "olmo7b", "gemma4e4b"]
ROWS = [("J0", "J-lens"), ("JPP0", "J++"), ("none", "A-lens alone"), ("J", "J-lens + A-lens"), ("JPP", "J++ + A-lens")]
PAIRS = [("JPP", "JPP0"), ("J", "J0"), ("none", "J0"), ("none", "JPP0")]
def boot(names, k, nb=5000):
    df = pd.concat([pd.read_csv(R / n / "readout_filtered.csv").assign(lens=n) for n in names])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[names].isna().any().any()
    evals = sorted(item.index.get_level_values(0).unique()); per = {e: item.loc[e][names].to_numpy() for e in evals}
    point = np.mean([per[e].mean(0) for e in evals], 0); rng = np.random.default_rng(0); b = np.empty((nb, len(names)))
    for i in range(nb): b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in evals], 0)
    return point, b
out = []
for t in TAGS:
    for kind in ("test", "v4"):
        names = {(r, lv): f"w1_{t}_{r}_{lv}_{kind}" for r, _ in ROWS for lv in ("f1", "f2", "f3")}
        if not all((R / n / "summary.json").exists() for n in names.values()): out.append(f"\n## {t} {kind}: INCOMPLETE"); continue
        S = {k: json.load(open(R / n / "summary.json")) for k, n in names.items()}
        out.append(f"\n## {t} {kind} (n items {S[('JPP','f1')]['n_items']}); top-10 / top-1")
        out.append("| row | f1 | f2 | f3 |\n|---|---|---|---|")
        for r, lab in ROWS: out.append(f"| {lab} | " + " | ".join(f"{100*S[(r,lv)]['pass10']:.1f} / {100*S[(r,lv)]['pass1']:.1f}" for lv in ("f1", "f2", "f3")) + " |")
        out.append("\n| difference (top-10; top-1) | f1 | f2 | f3 |\n|---|---|---|---|")
        res = {}
        for lv in ("f1", "f2", "f3"):
            ns = [names[(r, lv)] for r, _ in ROWS]
            for k in (10, 1):
                pt, b = boot(ns, k)
                for a, c in PAIRS:
                    i, j = [r for r, _ in ROWS].index(a), [r for r, _ in ROWS].index(c); d = b[:, i] - b[:, j]
                    res[(a, c, lv, k)] = f"{100*(pt[i]-pt[j]):+.1f} [{100*np.percentile(d,2.5):+.1f}, {100*np.percentile(d,97.5):+.1f}]"
        lab = dict(ROWS)
        for a, c in PAIRS: out.append(f"| {lab[a]} minus {lab[c]} | " + " | ".join(f"{res[(a,c,lv,10)]}; {res[(a,c,lv,1)]}" for lv in ("f1", "f2", "f3")) + " |")
txt = "\n".join(out); open("/workspace/o-lens-improvement/runs/w1/heldout_table.md", "w").write(txt); print(txt)
