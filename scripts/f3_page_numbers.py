"""All page numbers under filter f3 (held-out). Writes runs/f3_page.json and prints it."""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement/auto/runs")
TAGS = ["2b", "9b", "27b", "olmo7b", "gemma4e4b"]; ROWS = ["J0", "JPP0", "none", "J", "JPP"]
def S(n): return json.load(open(R / n / "summary.json"))
def boot(names, k, nb=5000):
    df = pd.concat([pd.read_csv(R / n / "readout_filtered.csv").assign(lens=n) for n in names])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index(); best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens"); assert not item[names].isna().any().any()
    ev = sorted(item.index.get_level_values(0).unique()); per = {e: item.loc[e][names].to_numpy() for e in ev}
    pt = np.mean([per[e].mean(0) for e in ev], 0); rng = np.random.default_rng(0); b = np.empty((nb, len(names)))
    for i in range(nb): b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in ev], 0)
    return pt, b
def diffs(N, pairs):
    names = list(N.values()); keys = list(N); out = {}
    for k in (10, 1):
        pt, b = boot(names, k)
        for a, c in pairs:
            i, j = keys.index(a), keys.index(c); d = b[:, i] - b[:, j]
            out[f"{a}-{c}@{k}"] = [round(100*(pt[i]-pt[j]), 1), round(100*np.percentile(d, 2.5), 1), round(100*np.percentile(d, 97.5), 1)]
    return out
P = {"w1": {}, "w2": {}, "w3": {}}
for t in TAGS:
    for kind in ("test", "v4"):
        N = {r: f"w1_{t}_{r}_f3_{kind}" for r in ROWS}; s = {r: S(n) for r, n in N.items()}
        P["w1"][f"{t}_{kind}"] = dict(n=s["JPP"]["n_items"], rows={r: [round(100*s[r]["pass10"], 1), round(100*s[r]["pass1"], 1)] for r in ROWS},
            task={r: {e: round(100*v, 1) for e, v in s[r]["per_task_pass10"].items()} for r in ("JPP0", "JPP")},
            d=diffs(N, [("JPP", "JPP0"), ("J", "J0"), ("J", "JPP0"), ("none", "J0"), ("none", "JPP0")]))
for t in ("9b", "2b"):
    for kind in ("test", "v4"):
        N = {"JPP0": f"w1_{t}_JPP0_f3_{kind}", "ridge": f"w1_{t}_none_f3_{kind}", "ce": f"w2ho_{t}_ce_none_f3_{kind}", "JPP+ridge": f"w1_{t}_JPP_f3_{kind}",
             "JPP+ce": f"w2ho_{t}_ce_JPP_f3_{kind}", "JPP+shuf": f"w2ho_{t}_shuf_JPP_f3_{kind}", "JPP+const": f"w2ho_{t}_const_JPP_f3_{kind}"}
        if not all((R / n / "summary.json").exists() for n in N.values()): continue
        s = {r: S(n) for r, n in N.items()}
        P["w2"][f"{t}_{kind}"] = dict(rows={r: [round(100*s[r]["pass10"], 1), round(100*s[r]["pass1"], 1)] for r in N},
            d=diffs(N, [("JPP+ce", "JPP+ridge"), ("ce", "ridge"), ("JPP+ce", "JPP0"), ("JPP+ce", "JPP+shuf"), ("JPP+ce", "JPP+const")]))
for kind in ("test", "v4"):
    N = {"JPP0": f"w1_9b_JPP0_f3_{kind}", "ridge": f"w1_9b_none_f3_{kind}", "JPP+ridge": f"w1_9b_JPP_f3_{kind}", "ce": f"w2ho_9b_ce_none_f3_{kind}", "JPP+ce": f"w2ho_9b_ce_JPP_f3_{kind}",
         "jacA5": f"w3ho_9b_jaA5_alone_f3_{kind}", "jacK1": f"w3ho_9b_jaK1_alone_f3_{kind}", "JPP+jacA5": f"w3ho_9b_jaA5_add_f3_{kind}", "JPP+jacK1": f"w3ho_9b_jaK1_add_f3_{kind}"}
    if not all((R / n / "summary.json").exists() for n in N.values()): continue
    s = {r: S(n) for r, n in N.items()}
    P["w3"][f"9b_{kind}"] = dict(rows={r: [round(100*s[r]["pass10"], 1), round(100*s[r]["pass1"], 1)] for r in N},
        d=diffs(N, [("JPP+jacA5", "JPP+ridge"), ("JPP+jacK1", "JPP+ridge"), ("JPP+jacA5", "JPP0"), ("JPP+jacA5", "JPP+ce"), ("jacA5", "ridge")]))
json.dump(P, open("/workspace/o-lens-improvement/runs/f3_page.json", "w"), indent=0); print("w1", len(P["w1"]), "w2", len(P["w2"]), "w3", len(P["w3"]))
