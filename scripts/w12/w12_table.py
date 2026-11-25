"""w12 analysis (hub python3, no torch): dev table with 95% bootstrap intervals, paired bootstraps, per-task,
fit-vs-benchmark. Bootstrap = scripts/w7/w7_boot.py (2000 draws, dev items resampled within task).
python3 scripts/w12/w12_table.py <tag>   -> prints markdown, writes runs/w12/table_<tag>.md"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w7"))
import w7_boot as B  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "2b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
out = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def ok(n):
    return (R / f"auto/runs/{n}/summary.json").exists()


def summ(n):
    return json.loads((R / f"auto/runs/{n}/summary.json").read_text())


def ci(n, nb=2000):
    A = B.items(n); byt = defaultdict(list)
    for k in A:
        byt[k[0]].append(k)
    rng = random.Random(0); v10, v1 = [], []
    for _ in range(nb):
        s = {t: [rng.choice(ks) for _ in ks] for t, ks in byt.items()}
        v10.append(sum(sum(A[k][0] for k in ks) / len(ks) for ks in s.values()) / len(s))
        v1.append(sum(sum(A[k][1] for k in ks) / len(ks) for ks in s.values()) / len(s))
    v10.sort(); v1.sort()
    return (v10[int(.025 * nb)], v10[int(.975 * nb) - 1]), (v1[int(.025 * nb)], v1[int(.975 * nb) - 1])


def chosen(n):
    p = R / f"runs/w12/evlogs/{n}.log"
    if not p.exists():
        return ""
    m = re.findall(r"\[w12 ([a-z]+)[^\]]*\] chosen ([0-9.]+)", p.read_text())
    m2 = re.findall(r"chosen lam ([0-9.]+)", p.read_text())
    return " ".join(f"{a}={b}" for a, b in m) + (f" lam={m2[-1]}" if m2 else "")


def fitstats(cfg):
    v = []
    for l in LAYERS:
        p = R / f"runs/w12/fits/{TAG}_{cfg}_L{l}.json"
        if not p.exists():
            return None
        v.append(json.loads(p.read_text()))
    m = lambda k: sum(x[k] for x in v) / len(v)  # noqa: E731
    return dict(val=m("val"), val0=m("val0"), train=m("train"), train0=m("train0"), val_T1=m("val_T1"),
                val_T16=m("val_T16"), rel=m("rel_dist"), relJ=m("rel_dist_J"), steps=sum(x["step"] for x in v) / len(v))


def row(label, stem, cfg=None):
    cells = [label]
    for f in ("f4", "f1"):
        n = f"{stem}_{f}"
        if not ok(n):
            cells += ["-", "-"]; continue
        s = summ(n); (a, b), (c, d) = ci(n)
        cells += [f"{s['pass10']:.3f} [{a:.2f},{b:.2f}]", f"{s['pass1']:.3f} [{c:.2f},{d:.2f}]"]
    fs = fitstats(cfg) if cfg else None
    cells += [f"{fs['train0']:.2f}→{fs['train']:.2f}", f"{fs['val0']:.2f}→{fs['val']:.2f}", f"{fs['relJ']:.2f}"] if fs else ["", "", ""]
    cells.append(chosen(f"{stem}_f4"))
    P("| " + " | ".join(cells) + " |")


ROWS = json.loads((R / f"runs/w12/rows_{TAG}.json").read_text())  # [[label, stem, cfg|null], ...]
P(f"## {TAG}: dev split, pass@10 / pass@1 [95% bootstrap CI], filters f4 and f1\n")
P("| row | f4 top-10 | f4 top-1 | f1 top-10 | f1 top-1 | train KL (init→best) | val KL (init→best) | ‖M−J‖/‖J‖ | chosen (f4) |")
P("|---|---|---|---|---|---|---|---|---|")
for lab, stem, cfg in ROWS:
    row(lab, stem, cfg)

PAIRS = json.loads((R / f"runs/w12/pairs_{TAG}.json").read_text())  # [[label, stemA, stemB], ...] -> B - A
for f in ("f4", "f1"):
    P(f"\n### {TAG}: paired bootstrap B − A (2000 draws, items resampled within task), filter {f}\n")
    P("| comparison | Δ top-10 [95% CI] | Δ top-1 [95% CI] |"); P("|---|---|---|")
    for lab, a, b in PAIRS:
        if ok(f"{a}_{f}") and ok(f"{b}_{f}"):
            (d10, l10, h10), (d1, l1, h1) = B.boot(f"{a}_{f}", f"{b}_{f}")
            P(f"| {lab} | {d10:+.3f} [{l10:+.3f},{h10:+.3f}] | {d1:+.3f} [{l1:+.3f},{h1:+.3f}] |")

P(f"\n### {TAG}: per-task top-10, filter f4\n")
P("| row | " + " | ".join(TASKS) + " |"); P("|---|" + "---|" * len(TASKS))
for lab, stem, cfg in ROWS:
    if ok(f"{stem}_f4"):
        t = summ(f"{stem}_f4")["per_task_pass10"]
        P(f"| {lab} | " + " | ".join(f"{t.get(k, float('nan')):.2f}" for k in TASKS) + " |")


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rk(x), rk(y); n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    c = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    return c / ((sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5 or 1)


P(f"\n### {TAG}: fit versus benchmark (maps alone; mean over layers of validation KL; dev f4 top-10)\n")
P("| map | val KL own target | val KL vs T1 | val KL vs T16 | dev f4 top-10 | dev f4 top-1 |"); P("|---|---|---|---|---|---|")
pts = []
for lab, stem, cfg in ROWS:
    if cfg and stem.endswith("_none") and ok(f"{stem}_f4"):
        fs = fitstats(cfg); s = summ(f"{stem}_f4")
        if fs:
            pts.append((cfg, fs, s))
            P(f"| {cfg} | {fs['val']:.3f} | {fs['val_T1']:.3f} | {fs['val_T16']:.3f} | {s['pass10']:.3f} | {s['pass1']:.3f} |")
if len(pts) > 3:
    for k in ("val", "val_T1", "val_T16"):
        x = [p[1][k] for p in pts]
        P(f"- Spearman(−{k}, dev top-10) = {spearman([-v for v in x], [p[2]['pass10'] for p in pts]):+.2f}; "
          f"(−{k}, dev top-1) = {spearman([-v for v in x], [p[2]['pass1'] for p in pts]):+.2f} (n={len(pts)}, shuffled controls included)")
    q = [p for p in pts if "shuf" not in p[0]]
    for k in ("val_T1",):
        P(f"- without shuffled controls: Spearman(−{k}, dev top-10) = "
          f"{spearman([-p[1][k] for p in q], [p[2]['pass10'] for p in q]):+.2f} (n={len(q)})")
(R / f"runs/w12/table_{TAG}.md").write_text("\n".join(out) + "\n")
