"""W2 held-out table (PREREG_v5 Amendment 3): cross-entropy A-lens (K=10) vs ridge A-lens, 2B/9B, test/v4, f1/f2."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement/auto/runs")
def boot(names, k, nb=5000):
    df = pd.concat([pd.read_csv(R / n / "readout_filtered.csv").assign(lens=n) for n in names])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index(); best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens"); assert not item[names].isna().any().any()
    ev = sorted(item.index.get_level_values(0).unique()); per = {e: item.loc[e][names].to_numpy() for e in ev}
    pt = np.mean([per[e].mean(0) for e in ev], 0); rng = np.random.default_rng(0); b = np.empty((nb, len(names)))
    for i in range(nb): b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in ev], 0)
    return pt, b
out = []
for t in ("9b", "2b"):
    for kind in ("test", "v4"):
        for lv in ("f1", "f2"):
            N = {"J++": f"w1_{t}_JPP0_{lv}_{kind}", "ridge alone": f"w1_{t}_none_{lv}_{kind}", "CE alone": f"w2ho_{t}_ce_none_{lv}_{kind}",
                 "J++ + ridge": f"w1_{t}_JPP_{lv}_{kind}", "J++ + CE": f"w2ho_{t}_ce_JPP_{lv}_{kind}",
                 "J++ + shuffled CE": f"w2ho_{t}_shuf_JPP_{lv}_{kind}", "J++ + constant CE": f"w2ho_{t}_const_JPP_{lv}_{kind}"}
            S = {k: json.load(open(R / n / "summary.json")) for k, n in N.items()}; names = list(N.values()); keys = list(N)
            out.append(f"\n## {t} {kind} {lv} (n {S['J++']['n_items']}): " + "; ".join(f"{k} {100*S[k]['pass10']:.1f}/{100*S[k]['pass1']:.1f}" for k in keys))
            P = [("J++ + CE", "J++ + ridge"), ("CE alone", "ridge alone"), ("J++ + CE", "J++"), ("J++ + CE", "J++ + shuffled CE"), ("J++ + CE", "J++ + constant CE")]
            r = {}
            for k in (10, 1):
                pt, b = boot(names, k)
                for a, c in P:
                    i, j = keys.index(a), keys.index(c); d = b[:, i] - b[:, j]
                    r[(a, c, k)] = f"{100*(pt[i]-pt[j]):+.1f} [{100*np.percentile(d,2.5):+.1f}, {100*np.percentile(d,97.5):+.1f}]"
            for a, c in P: out.append(f"  {a} minus {c}: top-10 {r[(a,c,10)]}; top-1 {r[(a,c,1)]}")
txt = "\n".join(out); open("/workspace/o-lens-improvement/runs/w2/heldout_table.md", "w").write(txt); print(txt)
