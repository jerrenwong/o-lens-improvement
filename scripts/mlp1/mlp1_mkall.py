"""from final_L20.json pick the all-layer configs: B, best C variant (C1024 / C4096 / C*-mean by val top-10 at L20),
D4096, and the controls P0 (bias-only) and L4096 (C without ReLU). Writes final_all.json and specs_all_s{0,1,2}.json."""
import json
S = "/workspace/o-lens-improvement/scripts/mlp1"
RES = f"{S}/results"
fin = json.load(open(f"{S}/final_L20.json"))
rows = [json.loads(x) for x in open(f"{RES}/runs_L20.jsonl")]


def group(r):  # as mlp1_pick.group
    g = r["kind"] + (str(r["width"]) if r["kind"] != "B" else "")
    return g + ("-mean" if r["data"] == "mean" else "") + ("-infer" if r["subset"] != "all" else "")


best = {}
for r in rows:
    if r["seed"] == 0 and (group(r) not in best or (r["val10"], r["val1"]) > (best[group(r)]["val10"], best[group(r)]["val1"])):
        best[group(r)] = r


def seedmean(g):  # mean val top-10 over the seeds of the group's chosen config
    b = best[g]
    v = [r["val10"] for r in rows if group(r) == g and all(r[k] == b[k] for k in
                                                           ("lr", "wd", "drop", "noise", "width", "data", "kind"))]
    return sum(v) / len(v)


cg = max([g for g in best if g.startswith("C") and not g.endswith("-infer")], key=seedmean)
print({g: round(seedmean(g), 4) for g in best})
out = {"B": fin["B"], "C": fin[cg], "D": fin["D4096"]}
for g in ("P0", "L4096", "Q0"):
    if g in fin:
        out[g] = fin[g]
print("C group chosen:", cg, json.dumps(out, indent=0))
json.dump(out, open(f"{S}/final_all.json", "w"), indent=1)
for seed in (0, 1, 2):
    json.dump([dict(s, seed=seed) for s in out.values()], open(f"{S}/specs_all_s{seed}.json", "w"), indent=1)
