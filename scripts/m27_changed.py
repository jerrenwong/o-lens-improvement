"""m27: per (item, intermediate) best rank over layers (filtered) for two runs; list items whose hit@10 changed, by task."""
import sys
import pandas as pd
R = "/workspace/o-lens-improvement/auto/runs"
a, b = sys.argv[1], sys.argv[2]
k = 10
best = {}
for n in (a, b):
    df = pd.read_csv(f"{R}/{n}/readout_filtered.csv")
    best[n] = df.groupby(["eval", "item", "intermediate"])["rank"].min()
m = pd.concat(best, axis=1).dropna()
m.columns = ["before", "after"]
ch = m[(m.before <= k) != (m.after <= k)]
for e, g in ch.groupby(level=0):
    up = g[g.after <= k]
    dn = g[g.after > k]
    print(f"== {e}: +{len(up)} gained, -{len(dn)} lost (of {int((m.loc[e].index.size))} item-intermediates)")
    for (_, item, inter), r in g.iterrows():
        print(f"   {'+' if r.after <= k else '-'} {item:28s} {inter!s:18s} {int(r.before):>6d} -> {int(r.after):<6d}")
print(f"total changed {len(ch)}; gained {(ch.after <= k).sum()}, lost {(ch.after > k).sum()}")
