"""dp1: tables from auto/runs/dp1_*/summary.json (+ per-layer hit shares from the scorer CSV: only the layer and
rank columns are read), runs/dp1/<set>/var/checks.json, diag.json and fit stats.
    python3 scripts/dp1/summarize.py > results/dp1_tables.md   (plain python, no torch)"""
import csv
import glob
import json
import os
from collections import defaultdict

R = "/workspace/o-lens-improvement"
VS = ["V0", "V1", "V2", "V2a", "V2b", "V3", "V4", "V5"]
SETS = [("2b_plain", "dp1_2bJ", ["released"], "2B J-lens (lrp none, K1, 64 prompts)"),
        ("9b_plain", "dp1_9bJ", ["stored"], "9B J-lens (lrp none, K1, 64 prompts)"),
        ("2b_jpp", "dp1_2bJPP", ["r1A", "released"], "2B J++ (rlens, K8, 128 prompts, weights on fit:0)")]
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")


def hits_per_layer(run):
    f = f"{R}/auto/runs/{run}/readout_filtered.csv"
    if not os.path.exists(f):
        return None
    n, h = defaultdict(int), defaultdict(int)
    with open(f) as fh:
        for row in csv.DictReader(fh):
            n[int(row["layer"])] += 1
            h[int(row["layer"])] += int(row["rank"]) <= 10
    return {l: h[l] / n[l] for l in sorted(n)}


def item_hits(run, k=10):
    """(task, item) -> share of its intermediates with best-over-layers rank <= k (filtered); keys only."""
    f = f"{R}/auto/runs/{run}/readout_filtered.csv"
    if not os.path.exists(f):
        return None
    best = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            key = (row["eval"], row["item"], row["intermediate"])
            best[key] = min(best.get(key, 10**9), int(row["rank"]))
    hits = defaultdict(list)
    for (e, it, _), r in best.items():
        hits[(e, it)].append(float(r <= k))
    return {key: sum(v) / len(v) for key, v in hits.items()}


def paired_ci(a, b, n=4000, seed=0):
    """pass@10 difference a-b (macro over tasks) with a 95% paired bootstrap over items within task."""
    import random
    rng = random.Random(seed)
    tasks = defaultdict(list)
    for key in a:
        tasks[key[0]].append(a[key] - b[key])
    stat = lambda groups: sum(sum(g) / len(g) for g in groups) / len(groups)
    obs = stat(list(tasks.values()))
    bs = sorted(stat([[g[rng.randrange(len(g))] for _ in g] for g in tasks.values()]) for _ in range(n))
    return 100 * obs, 100 * bs[int(0.025 * n)], 100 * bs[int(0.975 * n)]


for s, pre, refs, title in SETS:
    d = f"{R}/runs/dp1/{s}"
    if not os.path.exists(f"{d}/var/checks.json"):
        continue
    print(f"\n## {title}\n")
    stats = glob.glob(f"{d}/ckpt/*/fit/shard0of1_dp1_stats.json")
    if stats:
        st = json.load(open(stats[0]))
        print(f"content target share: {st['content_share']:.3f} ({st['targets_content']}/{st['targets_all']}), prompts {st['prompts']}\n")
    ck = json.load(open(f"{d}/var/checks.json"))["layers"]
    diag = json.load(open(f"{d}/diag.json")) if os.path.exists(f"{d}/diag.json") else None
    names = VS + refs
    summ = {}
    for v in names:
        p = f"{R}/auto/runs/{pre}_{v}/summary.json"
        if os.path.exists(p):
            summ[v] = json.load(open(p))
    base = summ.get("V0")
    print("| variant | dev p@10 | d vs V0 | p@1 | unf p@10 | " + " | ".join(TASKS) + " | tr/d range | next-tok top1 (FIT, mean over layers) | top10 non-content |")
    print("|" + "---|" * (9 + len(TASKS) - 4 + 4))
    for v in names:
        if v not in summ:
            continue
        x = summ[v]
        dv = f"{100 * (x['pass10'] - base['pass10']):+.1f}" if base else ""
        trs = [c["tr_over_d"][v] for c in ck.values()] if v in VS else (
            [r["tr_over_d"] for r in diag["lenses"][v].values()] if diag and v in diag["lenses"] else [])
        tr = f"{min(trs):.2f}..{max(trs):.2f}" if trs else ""
        nt = nc = ""
        if diag and v in diag["lenses"]:
            rr = list(diag["lenses"][v].values())
            nt = f"{100 * sum(r['top1_eq_next'] for r in rr) / len(rr):.1f}"
            nc = f"{100 * sum(r['top10_noncontent_share'] for r in rr) / len(rr):.1f}"
        print(f"| {v} | {100 * x['pass10']:.1f} | {dv} | {100 * x['pass1']:.1f} | {100 * x['pass10_unf']:.1f} | "
              + " | ".join(f"{100 * x['per_task_pass10'].get(t, float('nan')):.0f}" for t in TASKS)
              + f" | {tr} | {nt} | {nc} |")
    for k in (10, 1):
        h0 = item_hits(f"{pre}_V0", k)
        if h0:
            print(f"\npass@{k} difference vs V0, 95% paired bootstrap over dev items (within task):\n")
            for v in names[1:]:
                hv = item_hits(f"{pre}_{v}", k)
                if hv:
                    o, lo, hi = paired_ci(hv, h0)
                    print(f"- {v}: {o:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print("\nhits@10 per layer (share of (item, intermediate) rows with rank <= 10, filtered):\n")
    for v in names:
        h = hits_per_layer(f"{pre}_{v}")
        if h:
            print(f"- {v}: " + " ".join(f"L{l} {100 * x:.0f}" for l, x in h.items()))
    if diag:
        print("\nnext-token top-1 match per layer (FIT items, unfiltered):\n")
        for v, rr in diag["lenses"].items():
            print(f"- {v}: " + " ".join(f"L{l} {100 * r['top1_eq_next']:.0f}" for l, r in rr.items()))
        print("\ntop-10 non-content share per layer (FIT items, unfiltered):\n")
        for v, rr in diag["lenses"].items():
            print(f"- {v}: " + " ".join(f"L{l} {100 * r['top10_noncontent_share']:.0f}" for l, r in rr.items()))
    print("\nchecks per layer: max|S-I|, ||V1-V3||/||V0||, ||V2a-V2b||/||V2||, tr/d of D (same-position block):\n")
    for l, c in ck.items():
        print(f"- L{l}: {c['max_abs_S_minus_I']:.1e}, {c['relF_V1_V3']:.1e}, {c['relF_V2a_V2b']:.3f}, {c['tr_over_d_D']:.2f}")
    extra = glob.glob(f"{d}/v0_check.json")
    if extra:
        print("\nV0 reproduction: " + open(extra[0]).read())
