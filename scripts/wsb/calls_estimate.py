"""Rough call and token count of a summary-mode judge run over one readouts dir (no API call; stdlib only).

Usage: python3 calls_estimate.py READOUTS_DIR [--layers 24,32,40,48,56] [--families f1,f2]
`wsbench run dry_run=True` prints only the first digest prompt and no counts, so this estimates them:
  per-layer digests = items x layers (layered families); arithmetic_intermediates / chain_intermediates
  (numbers digests) digest all layers jointly = items x 1; plus one layered note-taker call per item
  (layered families); plus >= 1 judge call per item (several families use 3 draws, re-asks: x1-3).
  Hierarchical overflow chunks are not counted (more calls for very large items).
Input tokens of the digest calls ~ (bundle chars / 3.5) + ~1,500 system tokens per call. A lower bound for
input; output (digest JSON + low reasoning) is not estimable offline: measure it on the poetry trial.
"""
import argparse
import json
import os
from collections import defaultdict

JOINT = {"arithmetic_intermediates", "chain_intermediates"}
ap = argparse.ArgumentParser()
ap.add_argument("ro")
ap.add_argument("--layers", default="")
ap.add_argument("--families", default="")
a = ap.parse_args()
want_l = {int(x) for x in a.layers.split(",")} if a.layers else None
fams = a.families.split(",") if a.families else sorted(f[:-6] for f in os.listdir(a.ro) if f.endswith(".jsonl"))
print("| family | items | layers | per-layer digests | layered | judge (>=) | calls (>=) | bundle Mtok | in Mtok (>=) |")
print("|---|---|---|---|---|---|---|---|---|")
T = defaultdict(float)
for fam in fams:
    items, layers, chars = set(), set(), 0
    with open(os.path.join(a.ro, fam + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if want_l is not None and r["layer"] not in want_l:
                continue
            items.add(r["id"]); layers.add(r["layer"])
            chars += 20 + sum(len(t) + 3 for t in r["tokens"])
    n, L = len(items), len(layers)
    per = n if fam in JOINT else n * L
    lay = 0 if fam in JOINT or L == 1 else n
    calls = per + lay + n
    btok = chars / 3.5
    intok = btok * (2 if lay else 1) + 1500 * calls
    for k, v in (("items", n), ("per", per), ("lay", lay), ("judge", n), ("calls", calls), ("btok", btok), ("in", intok)):
        T[k] += v
    print(f"| {fam} | {n} | {L} | {per} | {lay} | {n} | {calls} | {btok / 1e6:.2f} | {intok / 1e6:.1f} |")
print(f"| **total** | {int(T['items'])} | | {int(T['per'])} | {int(T['lay'])} | {int(T['judge'])} | {int(T['calls'])} | "
      f"{T['btok'] / 1e6:.1f} | {T['in'] / 1e6:.1f} |")
