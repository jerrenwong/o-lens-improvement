"""w11 analysis (hub python3, no torch). Scores from auto/runs/<name>/readout_filtered.csv with evaluate.py's rule;
paired bootstrap = scripts/w7/w7_boot.py (2000 draws, dev items resampled within task, B - A).
  python3 scripts/w11/w11_table.py row <name>...                 top-10 / top-1 (+ unfiltered) and per-task
  python3 scripts/w11/w11_table.py boot <A> <B> [<A> <B>...]      B - A, 95% CI
  python3 scripts/w11/w11_table.py cover <tag> <nmin> <A> <B>     scores and B - A on covered / uncovered concepts"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w7"))
import w7_boot as B  # noqa: E402

TASKS = ("association", "multihop", "multilingual", "poetry", "typo")


def concepts(name):
    best = {}
    with open(R / f"auto/runs/{name}/readout_filtered.csv") as f:
        for r in csv.DictReader(f):
            k = (r["eval"], r["item"], r["intermediate"]); rk = int(r["rank"])
            best[k] = min(best.get(k, rk), rk)
    return best


def items_from(best, keep=None):
    it = defaultdict(list)
    for (e, i, c), rk in best.items():
        if keep is None or keep(e, i, c):
            it[(e, i)].append(rk)
    return {k: (sum(r <= 10 for r in v) / len(v), sum(r <= 1 for r in v) / len(v)) for k, v in it.items()}


def score_tasks(d):
    byt = defaultdict(list)
    for k, v in d.items():
        byt[k[0]].append(v)
    t10 = {t: sum(x[0] for x in v) / len(v) for t, v in byt.items()}
    t1 = {t: sum(x[1] for x in v) / len(v) for t, v in byt.items()}
    return sum(t10.values()) / len(t10), sum(t1.values()) / len(t1), t10, t1


def boot_d(A, Bd, n=2000, seed=0):
    assert A.keys() == Bd.keys()
    byt = defaultdict(list)
    for k in A:
        byt[k[0]].append(k)
    def sc(D, s):
        return [sum(sum(D[k][j] for k in ks) / len(ks) for ks in s.values()) / len(s) for j in (0, 1)]
    d0 = [b - a for a, b in zip(sc(A, byt), sc(Bd, byt))]
    rng = random.Random(seed); ds = []
    for _ in range(n):
        s = {t: [rng.choice(ks) for _ in ks] for t, ks in byt.items()}
        a, b = sc(A, s), sc(Bd, s)
        ds.append((b[0] - a[0], b[1] - a[1]))
    out = []
    for j in (0, 1):
        v = sorted(x[j] for x in ds)
        out.append((d0[j], v[int(0.025 * n)], v[int(0.975 * n) - 1]))
    return out


def summ(name):
    return json.loads((R / f"auto/runs/{name}/summary.json").read_text())


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "row":
        for n in a:
            s = summ(n)
            p10, p1, t10, t1 = score_tasks(items_from(concepts(n)))
            assert abs(p10 - s["pass10"]) < 1e-9 and abs(p1 - s["pass1"]) < 1e-9
            tasks = " ".join(f"{t[:5]} {t10.get(t, float('nan')):.3f}/{t1.get(t, float('nan')):.3f}" for t in TASKS)
            print(f"| {n} | {p10:.3f} / {p1:.3f} | {s['pass10_unf']:.3f} / {s['pass1_unf']:.3f} | {tasks} |")
    elif cmd == "boot":
        for x, y in zip(a[::2], a[1::2]):
            (d10, l10, h10), (d1, l1, h1) = B.boot(x, y)
            print(f"| {y} - {x} | {d10:+.3f} [{l10:+.3f}, {h10:+.3f}] | {d1:+.3f} [{l1:+.3f}, {h1:+.3f}] |")
    elif cmd == "cover":
        tag, nmin, x, y = a[0], a[1], a[2], a[3]
        flags = json.loads((R / f"runs/w11/cover_{tag}.json").read_text())[nmin]["dev_flags"]
        cx, cy = concepts(x), concepts(y)
        for lab, want in (("covered", True), ("uncovered", False)):
            keep = lambda e, i, c: flags[f"{e}|{i}|{c}"] == want  # noqa: E731
            A, Bd = items_from(cx, keep), items_from(cy, keep)
            if not A:
                print(lab, "none"); continue
            pa, pb = score_tasks(A), score_tasks(Bd)
            (d10, l10, h10), (d1, l1, h1) = boot_d(A, Bd)
            nc = sum(1 for k in cx if flags[f"{k[0]}|{k[1]}|{k[2]}"] == want)
            print(f"| {lab} ({nc} concepts, {len(A)} items) | {pa[0]:.3f} / {pa[1]:.3f} | {pb[0]:.3f} / {pb[1]:.3f} | "
                  f"{d10:+.3f} [{l10:+.3f}, {h10:+.3f}] | {d1:+.3f} [{l1:+.3f}, {h1:+.3f}] |")
