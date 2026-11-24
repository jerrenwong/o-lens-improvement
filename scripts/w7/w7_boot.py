"""W7: dev table + paired bootstrap from auto/runs/<name>/readout_filtered.csv (evaluate.py's pass_at_k rule:
per concept best rank over layers; item = mean over its concepts of rank<=k; task = mean over items; score = mean over
tasks). Bootstrap: 2000 draws, resample dev items with replacement within each task, same draw for both runs.
Usage: python w7_boot.py table <prefix> <variant>...          (prints top-10 / top-1 / lam per tag, row, filter)
       python w7_boot.py boot <runA> <runB> [...pairs]         (B - A)"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

R = Path("/workspace/o-lens-improvement")


def items(name):
    best = {}
    with open(R / f"auto/runs/{name}/readout_filtered.csv") as f:
        for r in csv.DictReader(f):
            k = (r["eval"], r["item"], r["intermediate"]); rk = int(r["rank"])
            best[k] = min(best.get(k, rk), rk)
    it = defaultdict(list)
    for (e, i, _), rk in best.items():
        it[(e, i)].append(rk)
    return {k: (sum(r <= 10 for r in v) / len(v), sum(r <= 1 for r in v) / len(v)) for k, v in it.items()}


def score(d, keys_by_task):
    t10 = [sum(d[k][0] for k in ks) / len(ks) for ks in keys_by_task.values()]
    t1 = [sum(d[k][1] for k in ks) / len(ks) for ks in keys_by_task.values()]
    return sum(t10) / len(t10), sum(t1) / len(t1)


def boot(a, b, n=2000, seed=0):
    A, B = items(a), items(b)
    assert A.keys() == B.keys(), (a, b)
    byt = defaultdict(list)
    for k in A:
        byt[k[0]].append(k)
    d0 = [y - x for x, y in zip(score(A, byt), score(B, byt))]
    rng = random.Random(seed); ds = []
    for _ in range(n):
        s = {t: [rng.choice(ks) for _ in ks] for t, ks in byt.items()}
        # duplicate keys allowed: score over lists
        sa = [sum(A[k][j] for k in ks) / len(ks) for j in (0, 1) for ks in s.values()]
        sb = [sum(B[k][j] for k in ks) / len(ks) for j in (0, 1) for ks in s.values()]
        nt = len(s)
        ds.append(((sum(sb[:nt]) - sum(sa[:nt])) / nt, (sum(sb[nt:]) - sum(sa[nt:])) / nt))
    out = []
    for j in (0, 1):
        v = sorted(x[j] for x in ds)
        out.append((d0[j], v[int(0.025 * n)], v[int(0.975 * n) - 1]))
    return out


def lam(name):
    p = R / f"runs/w7/logs/{name}.log"
    m = re.findall(r"chosen lam ([0-9.]+)", p.read_text()) if p.exists() else []
    return m[-1] if m else "-"


if __name__ == "__main__":
    if sys.argv[1] == "table":
        pre, vars_ = sys.argv[2], sys.argv[3:]
        for tag in ("2b", "9b"):
            print(f"\n{tag} | variant | f4 alone | f4 J++ + A | f1 alone | f1 J++ + A   (top-10 / top-1 (lam))")
            for v in vars_:
                cells = []
                for lv in ("f4", "f1"):
                    for row in (("JPP0",) if v == "JPP0" else ("none", "JPP")):
                        n = f"{pre}_{tag}_{v}_{lv}" if v == "JPP0" else f"{pre}_{tag}_{v}_{row}_{lv}"
                        sj = R / f"auto/runs/{n}/summary.json"
                        if sj.exists():
                            s = json.loads(sj.read_text())
                            cells.append(f"{s['pass10']:.3f} / {s['pass1']:.3f}" + (f" ({lam(n)})" if row == "JPP" else ""))
                        else:
                            cells.append("n/a")
                print(f"{tag} | {v} | " + " | ".join(cells))
    else:
        a = sys.argv[2:]
        for x, y in zip(a[::2], a[1::2]):
            (d10, l10, h10), (d1, l1, h1) = boot(x, y)
            print(f"{y} - {x}: top-10 {d10:+.3f} [{l10:+.3f}, {h10:+.3f}]  top-1 {d1:+.3f} [{l1:+.3f}, {h1:+.3f}]")
