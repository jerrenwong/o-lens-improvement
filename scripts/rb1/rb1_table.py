"""rb1 table: dev scores (auto/runs/rb1_*_<tag>/summary.json), chosen lam + FIT score (eval logs /root/rb1), and the
label-free curves (.hf27/rb1/fit_<tag>.json, p3v_<tag>.json). Usage: python rb1_table.py <tag>  (run on the pod)."""
import json
import re
import statistics as st
import sys

R = "/workspace/o-lens-improvement"
TAG = sys.argv[1]
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")


def summ(name):
    try:
        return json.load(open(f"{R}/auto/runs/{name}/summary.json"))
    except FileNotFoundError:
        return None


def lam_fit(name):
    try:
        txt = open(f"/root/rb1/eval_{name}.log").read()
    except FileNotFoundError:
        return None, None
    m = re.search(r"chosen lam ([0-9.]+)", txt)
    if not m:
        return None, None
    lam = m.group(1)
    f = re.search(r"lam" + re.escape(lam) + r"\s+p10 ([0-9.]+) p1 ([0-9.]+)", txt)
    return float(lam), (float(f.group(1)) + float(f.group(2))) if f else None


def row(label, name, alone_name=None):
    s, a = summ(name), summ(alone_name) if alone_name else None
    lam, fit = lam_fit(name)
    out = dict(label=label, lam=lam, fit=fit)
    if s:
        out.update(p10=s["pass10"], p1=s["pass1"], p10u=s["pass10_unf"], task=s["per_task_pass10"])
    if a:
        out.update(A10=a["pass10"], A1=a["pass1"], A10u=a["pass10_unf"])
    return out


rows = [row("J++ (unit)", f"rb1_jppu_{TAG}"), row("L stored", f"rb1_Lstored_{TAG}", f"rb1_Lstored_A_{TAG}"),
        row("P3v stored", f"rb1_P3vstored_{TAG}", f"rb1_P3vstored_A_{TAG}")]
V = [f"r{n}_{k}" for n in (300, 1000, 3000, 10000) for k in range(3)] + ["hA", "hB", "rall"] + \
    [f"p{n}_{k}" for n in (1000, 10000) for k in range(2)]
for v in V:
    rows.append(row(v, f"rb1_{v}_{TAG}", f"rb1_{v}_A_{TAG}"))


def f(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}"


print(f"== {TAG}: dev (x100). lam / FIT(p1+p10) / L p10 p1 unf-p10 / ALONE p10 p1 unf-p10 / per-task L p10")
for r in rows:
    t = " ".join(f"{k[:5]} {f(r['task'].get(k), 0)}" for k in TASKS) if "task" in r else ""
    print(f"{r['label']:12s} lam {r['lam']!s:5s} FIT {f(r['fit'])} | L {f(r.get('p10'))} {f(r.get('p1'))} {f(r.get('p10u'))} "
          f"| A {f(r.get('A10'))} {f(r.get('A1'))} {f(r.get('A10u'))} | {t}")

jpp = rows[0]
print("\n== grouped by n (mean [min, max] over subsets); gain = L - J++ (unit)")
groups = [(f"ridge {n}", [f"r{n}_{k}" for k in range(3)]) for n in (300, 1000, 3000, 10000)] + \
         [("ridge halves", ["hA", "hB"]), ("ridge all", ["rall"])] + \
         [(f"P3v {n}", [f"p{n}_{k}" for k in range(2)]) for n in (1000, 10000)] + [("P3v all", None)]
byl = {r["label"]: r for r in rows}
for g, vs in groups:
    rs = [byl[v] for v in vs] if vs else [byl["P3v stored"]]
    cells = []
    for k in ("p10", "p1", "p10u", "A10", "A1"):
        xs = [r[k] for r in rs if k in r]
        if not xs:
            cells.append(f"{k} -"); continue
        cells.append(f"{k} {f(st.mean(xs))} [{f(min(xs))},{f(max(xs))}]")
    gains = [r["p10"] - jpp["p10"] for r in rs if "p10" in r and "p10" in jpp]
    gains1 = [r["p1"] - jpp["p1"] for r in rs if "p1" in r and "p1" in jpp]
    lams = [r["lam"] for r in rs]
    print(f"{g:13s} " + "  ".join(cells) + f"  | gain p10 {' '.join(f(x) for x in gains)}  p1 {' '.join(f(x) for x in gains1)}"
          f" | lam {lams}")

print("\n== label-free (fit-script TEST prefixes, mean over layers): ridge")
F = json.load(open(f"{R}/.hf27/rb1/fit_{TAG}.json"))
print("repro of stored map (all layers bitwise W, c, mx, my):",
      all(v["W_bitwise"] and v["c_equal"] and v["mx_equal"] and v["my_equal"] for v in F["repro"].values()))
for g, vs in groups[:6]:
    ms = [F["mean"][v] for v in vs]
    print(f"{g:13s} n={F['n'][vs[0]]:5d} " + " ".join(
        f"{k} {st.mean(m[k] for m in ms):.4f}" + (f"±{(max(m[k] for m in ms) - min(m[k] for m in ms)) / 2:.4f}" if len(ms) > 1 else "")
        for k in ("val10", "a10", "a1", "a1u", "mse", "nw1")))
try:
    P = json.load(open(f"{R}/.hf27/rb1/p3v_{TAG}.json"))
    print("\n== label-free P3v (test prefixes, mean over layers): held-out KL to p over KEEP, Spearman, top-1 kept, a10, a1; "
          "val KL at tau=1 (full vocab)")
    for nm, r in P.items():
        m = r["heldout_mean"]
        steps = [v["step"] for v in r["train"].values()] if r["train"] else []
        print(f"{nm:9s} n={r['n']:5d} KL {m['kl']:.3f} Sp {m['spearman']:.3f} top1 {m['top1']:.3f} a10 {m['a10']:.4f} "
              f"a1 {m['a1']:.4f} valKL {m['val_kl_tau1']:.3f} best steps {steps}")
except FileNotFoundError:
    pass
