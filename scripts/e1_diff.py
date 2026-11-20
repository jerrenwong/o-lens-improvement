"""e1: per-(item, intermediate) best rank change between two dev runs (auto/runs/<a>, <b>), tasks given."""
import sys
import pandas as pd
R = "/workspace/o-lens-improvement/auto/runs"
a, b = sys.argv[1], sys.argv[2]
tasks = sys.argv[3].split(",")
def best(n):
    d = pd.read_csv(f"{R}/{n}/readout_filtered.csv")
    return d.groupby(["eval", "item", "intermediate"])["rank"].min()
m = pd.concat([best(b).rename("before"), best(a).rename("after")], axis=1).reset_index()
m = m[m["eval"].isin(tasks)]
ch = m[(m.before <= 10) != (m.after <= 10)]
for e in tasks:
    s = ch[ch["eval"] == e]
    print(f"\n{e}: gained {int(((s.after <= 10)).sum())}  lost {int(((s.before <= 10)).sum())}  (of {int((m['eval']==e).sum())} intermediates)")
    for r in s.sort_values("after").itertuples():
        print(f"  {'+' if r.after <= 10 else '-'} {r.item:32s} {r.intermediate:18s} {r.before:>6d} -> {r.after}")
