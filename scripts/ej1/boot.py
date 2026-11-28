"""ej1 paired bootstrap on the DEV readouts (auto/runs/<name>/readout_filtered.csv): evaluator metric (per (item,
intermediate) best rank over layers; per item share of intermediates hit; mean per task; mean over tasks), 2000 draws,
items resampled within task, paired (same draws for both lenses).
    python3 scripts/ej1/boot.py BASE_RUN NEW_RUN [NEW_RUN ...] [--k 10]
"""
import csv
import random
import sys
from collections import defaultdict

R = "/workspace/o-lens-improvement/auto/runs"


def per_item(run, k, kind="filtered"):
    best = {}
    for r in csv.DictReader(open(f"{R}/{run}/readout_{kind}.csv")):
        key = (r["eval"], r["item"], r["intermediate"])
        rk = int(r["rank"])
        best[key] = min(best.get(key, rk), rk)
    hits = defaultdict(list)
    for (e, it, _), rk in best.items():
        hits[(e, it)].append(rk <= k)
    return {key: sum(v) / len(v) for key, v in hits.items()}


def metric(d, items_by_task):
    return sum(sum(d[x] for x in its) / len(its) for its in items_by_task.values()) / len(items_by_task)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ks = [1, 10]
    base, news = args[0], args[1:]
    for k in ks:
        A = per_item(base, k)
        tasks = defaultdict(list)
        for key in sorted(A):
            tasks[key[0]].append(key)
        rng = random.Random(0)
        draws = [{e: [rng.choice(v) for _ in v] for e, v in tasks.items()} for _ in range(2000)]
        pa = metric(A, tasks)
        for n in news:
            B = per_item(n, k)
            assert set(B) == set(A), (n, len(B), len(A))
            pb = metric(B, tasks)
            ds = sorted(metric(B, dr) - metric(A, dr) for dr in draws)
            pt = {e: (sum(B[x] for x in v) - sum(A[x] for x in v)) / len(v) for e, v in tasks.items()}
            print(f"top{k:<2} {n:34s} vs {base:28s} {100 * pa:5.1f} -> {100 * pb:5.1f}  d {100 * (pb - pa):+5.1f} "
                  f"[{100 * ds[50]:+5.1f},{100 * ds[1949]:+5.1f}]  per task " +
                  " ".join(f"{e[:5]} {100 * v:+.0f}" for e, v in pt.items()))


if __name__ == "__main__":
    main()
