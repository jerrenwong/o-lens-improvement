"""Layer-20 tables from results/runs_L20.jsonl: (1) every run (grid), (2) per group the val-chosen config x seeds.
Usage: python3 mlp1_table20.py > results/table_L20.md"""
import json, statistics as st
RES = "/workspace/o-lens-improvement/scripts/mlp1/results"
rows = [json.loads(x) for x in open(f"{RES}/runs_L20.jsonl")]
seen = {}
for r in rows:
    seen[r["name"]] = r  # last write wins (no duplicates expected)
rows = list(seen.values())


def group(r):
    g = r["kind"] + (str(r["width"]) if r["kind"] != "B" else "")
    return g + ("-mean" if r["data"] == "mean" else "") + ("-infer" if r["subset"] != "all" else "")


def cfg(r):
    return f"lr {r['lr']:g} wd {r['wd']:g} drop {r['drop']:g}" + (f" noise {r['noise']:g}" if r["data"] == "mean" else "")


print("## all seed-0 runs (layer 20)\n")
print("| group | config | val top10 | test top10 | test top1 | train-word top10 | best step / steps |")
print("|---|---|---|---|---|---|---|")
for r in sorted([r for r in rows if r["seed"] == 0], key=lambda r: (group(r), -r["val10"])):
    print(f"| {group(r)} | {cfg(r)} | {r['val10']:.4f} | {r['test10']:.4f} | {r['test1']:.4f} | {r['train10']:.4f} | "
          f"{r['best_step']}/{r['steps']} |")
best = {}
for r in rows:
    if r["seed"] == 0 and (group(r) not in best or (r["val10"], r["val1"]) > (best[group(r)]["val10"], best[group(r)]["val1"])):
        best[group(r)] = r
print("\n## val-chosen config per group, seeds (mean, sd over seeds)\n")
print("| group | config | seeds | val top10 | test top10 | test top1 | train-word top10 | best steps | mix test top10 (lam .1/.3/1) |")
print("|---|---|---|---|---|---|---|---|---|")
for g, b in sorted(best.items()):
    rs = [r for r in rows if group(r) == g and cfg(r) == cfg(b)]
    rs.sort(key=lambda r: r["seed"])

    def ms(k):
        v = [r[k] for r in rs]
        return f"{st.mean(v):.4f}" + (f" ± {st.stdev(v):.4f}" if len(v) > 1 else "")
    mix = "/".join(f"{st.mean([r[f'mixtest{lam}'] for r in rs]):.4f}" for lam in (0.1, 0.3, 1.0))
    print(f"| {g} | {cfg(b)} | {len(rs)} | {ms('val10')} | {ms('test10')} | {ms('test1')} | {ms('train10')} | "
          f"{','.join(str(r['best_step']) for r in rs)} | {mix} |")
