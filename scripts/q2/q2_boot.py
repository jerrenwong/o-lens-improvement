"""q2: paired bootstrap on the DEV split between two evaluator runs (auto/runs/<name>/readout_filtered.csv), metric =
auto/evaluate.py pass@k (per (item, intermediate) best rank over layers; per item share of intermediates hit; mean per
task; mean over tasks). Items resampled with replacement WITHIN task (same draw for both lenses), 2000 draws, seed 0.
    python3 scripts/q2/q2_boot.py <runA> <runB> [<runB2> ...] [--k 10] [--draws 2000]
Prints, per B: pass@10 and pass@1 of A and B, delta, 95% interval, share of draws with delta <= 0 (and >= 0)."""
import argparse
import csv
import random
from collections import defaultdict

R = "/workspace/o-lens-improvement"


def items(run):
    best = {}
    for r in csv.DictReader(open(f"{R}/auto/runs/{run}/readout_filtered.csv")):
        key = (r["eval"], r["item"], r["intermediate"])
        rk = int(r["rank"])
        best[key] = min(best.get(key, rk), rk)
    per = defaultdict(list)
    for (e, it, _), rk in best.items():
        per[(e, it)].append(rk)
    return per


def score(per, keys_by_task, k):
    return sum(sum(sum(r <= k for r in per[key]) / len(per[key]) for key in ks) / len(ks)
               for ks in keys_by_task.values()) / len(keys_by_task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    A = items(a.runs[0])
    by_task = defaultdict(list)
    for key in sorted(A):
        by_task[key[0]].append(key)
    rng = random.Random(0)
    draws = [{t: [rng.choice(ks) for _ in ks] for t, ks in by_task.items()} for _ in range(a.draws)]
    print(f"A = {a.runs[0]} ({sum(map(len, by_task.values()))} dev items)")
    print("| B | k | A | B | delta | 95% interval | P(delta<=0) | P(delta>=0) |")
    print("|---|---|---|---|---|---|---|---|")
    for run in a.runs[1:]:
        B = items(run)
        assert set(B) == set(A), run
        for k in (10, 1):
            sa, sb = score(A, by_task, k), score(B, by_task, k)
            ds = sorted(score(B, d, k) - score(A, d, k) for d in draws)
            lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
            print(f"| {run} | {k} | {sa:.3f} | {sb:.3f} | {sb - sa:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                  f"{sum(x <= 0 for x in ds) / len(ds):.3f} | {sum(x >= 0 for x in ds) / len(ds):.3f} |", flush=True)


if __name__ == "__main__":
    main()
