"""mlp1 hyperparameter picking at one layer, on VALIDATION words only (val top-10, ties val top-1).
  stage2: per group (B, C1024, C4096, D4096) best stage-1 lr -> + input dropout 0.1, + wd 1e-2; C (both widths) per-word
          class-mean training with noise sigma in {0, 0.5, 1.0} at its best lr.
  stage3: per group best config (seed 0) -> seeds 1, 2.
  final:  writes final.json {group: spec without seed}.
Usage: mlp1_pick.py LAYER stage2|stage3|final OUT.json"""
import json, sys
from collections import defaultdict

RES = "/workspace/o-lens-improvement/scripts/mlp1/results"
l, stage, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(x) for x in open(f"{RES}/runs_L{l}.jsonl")]
KEYS = ["kind", "width", "lr", "wd", "drop", "data", "noise", "subset"]


def group(r):
    g = r["kind"] + (str(r["width"]) if r["kind"] != "B" else "")
    return g + ("-mean" if r["data"] == "mean" else "") + ("-infer" if r["subset"] != "all" else "")


best = {}
for r in rows:
    if r["seed"] != 0:
        continue
    g = group(r)
    if g not in best or (r["val10"], r["val1"]) > (best[g]["val10"], best[g]["val1"]):
        best[g] = r
spec = {g: {k: r[k] for k in KEYS} for g, r in best.items()}
specs = []
if stage == "stage2":
    for g in ["B", "C1024", "C4096", "D4096"]:
        s = spec[g]
        specs += [dict(s, drop=0.1), dict(s, wd=1e-2)]
        tried = sorted({r["lr"] for r in rows if group(r) == g and r["drop"] == 0 and r["wd"] == 0})
        if s["lr"] == tried[-1]:  # best lr at the grid edge -> extend the grid one step
            specs.append(dict(s, lr={3e-4: 1e-3, 1e-3: 3e-3, 1e-4: 3e-4}[s["lr"]]))
        elif s["lr"] == tried[0]:
            specs.append(dict(s, lr={3e-5: 1e-5, 1e-5: 3e-6, 1e-4: 3e-5}[s["lr"]]))
        if g.startswith("C"):
            specs += [dict(s, data="mean", noise=z) for z in (0.0, 0.5, 1.0)]
    # linear control: C4096 without the ReLU (same parametrisation / lr scaling), its own lr grid
    specs += [dict(spec["C4096"], kind="L", lr=lr, drop=0.0, wd=0.0) for lr in (1e-5, 3e-5, 1e-4, 3e-4)]
    # bias-only control: ridge frozen, trained output shift
    specs += [dict(kind="P", width=0, lr=lr, wd=0.0, drop=0.0, data="sent", noise=0.0, subset="all") for lr in (1e-4, 1e-3, 1e-2)]
elif stage == "stage3":
    for g, s in spec.items():
        if g.endswith("-infer"):
            continue
        specs += [dict(s, seed=1), dict(s, seed=2)]
elif stage == "final":
    specs = {g: s for g, s in spec.items()}
print(json.dumps({g: [best[g]["name"], best[g]["val10"]] for g in best}, indent=0))
json.dump(specs, open(out, "w"), indent=1)
