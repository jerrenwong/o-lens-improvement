"""W5: table of native (no-API) scores per arm and family from <dir>/<arm>/<family>/results.json,
plus the lambda-sweep view (score at each lambda, fit-chosen lambda 0.5, benchmark-best lambda, regret).
Diagnostic only: the benchmark-best lambda is never adopted. Usage: python summarize.py <dir> [out.md]"""
import json
import os
import sys

D = sys.argv[1]
FAMS = ["multilingual", "multilingual_typo", "typo", "model_specific_directed_modulation"]
LAM_FIT = 0.5
arms = sorted(a for a in os.listdir(D) if os.path.isdir(f"{D}/{a}"))
res = {}
for a in arms:
    for f in FAMS:
        p = f"{D}/{a}/{f}/results.json"
        if os.path.exists(p):
            n = json.load(open(p))
            res[a, f] = (n["numbers"]["value"], n["numbers"]["ci95"], n["n_items"],
                         n["numbers"].get("extras", {}).get("any_hit_rate"))


def lam_of(a):
    if a == "jpp":
        return 0.0
    if a.startswith("jpp_a"):
        return float(a[5:])
    if a == "alens":
        return float("inf")
    return None


order = sorted([a for a in arms if lam_of(a) is not None], key=lam_of) + [a for a in arms if lam_of(a) is None]
L = []
L.append("| arm | lam | " + " | ".join(f"{f} (n)" for f in FAMS) + " | macro |")
L.append("|---|---|" + "---|" * (len(FAMS) + 1))
macro = {}
for a in order:
    cells, vals = [], []
    for f in FAMS:
        if (a, f) in res:
            v, ci, n, anyh = res[a, f]
            cells.append(f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({n})")
            vals.append(v)
        else:
            cells.append("-")
    macro[a] = sum(vals) / len(vals) if len(vals) == len(FAMS) else None
    lam = lam_of(a)
    L.append(f"| {a} | {'' if lam is None else lam:} | " + " | ".join(cells) + f" | {macro[a] if macro[a] is None else round(macro[a], 4)} |")
L.append("")
sweep = [a for a in order if lam_of(a) is not None]
L.append("| family | score at fit lam 0.5 | best lam on bench | best score | regret |")
L.append("|---|---|---|---|---|")
fit_arm = "jpp_a0.5"
for f in FAMS + ["macro"]:
    sc = {a: (macro[a] if f == "macro" else res.get((a, f), (None,))[0]) for a in sweep}
    sc = {a: v for a, v in sc.items() if v is not None}
    if fit_arm not in sc:
        continue
    best = max(sc.values())
    best_arms = [a for a in sweep if sc.get(a) == best]
    L.append(f"| {f} | {sc[fit_arm]:.3f} | {', '.join(str(lam_of(a)) for a in best_arms)} | {best:.3f} | {best - sc[fit_arm]:.3f} |")
txt = "\n".join(L)
print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(txt + "\n")

# ---- unit-level diagnostic (runner's per-item unit_hit; the headline needs every required unit, e.g. the
# language unit, which no token lens here gets): concept / native-correction hit at any judged layer.
import random  # noqa: E402

UNITS = {"multilingual": "concept", "multilingual_typo": "correction_native"}
U = ["", "| arm | lam | " + " | ".join(f"{f}:{u} [95% CI]" for f, u in UNITS.items()) + " | mean | diff vs lam 0.5 [95% paired CI] |",
     "|---|---|" + "---|" * (len(UNITS) + 2)]
hits = {}
for a in order:
    for f, u in UNITS.items():
        p = f"{D}/{a}/{f}/results.json"
        if os.path.exists(p):
            rows = json.load(open(p))["rows"]
            hits[a, f] = {r["id"]: float(bool(r["unit_hit"].get(u))) for r in rows}
rng = random.Random(0)
keys = {f: sorted(hits.get((fit_arm, f), {})) for f in UNITS}
draws = [{f: [rng.choice(keys[f]) for _ in keys[f]] for f in UNITS} for _ in range(2000)]


def mean_over(a, sample):
    return sum(sum(hits[a, f][i] for i in sample[f]) / len(sample[f]) for f in UNITS) / len(UNITS)


for a in order:
    if not all((a, f) in hits for f in UNITS):
        continue
    cells = []
    for f in UNITS:
        h = hits[a, f]
        ks = keys[f]
        bs = sorted(sum(h[i] for i in d[f]) / len(ks) for d in draws)
        cells.append(f"{sum(h.values()) / len(h):.3f} [{bs[50]:.3f}, {bs[1949]:.3f}]")
    full = {f: keys[f] for f in UNITS}
    m = mean_over(a, full)
    if (fit_arm, UNITS and list(UNITS)[0]) in hits and a != fit_arm:
        dd = sorted(mean_over(a, d) - mean_over(fit_arm, d) for d in draws)
        diff = f"{m - mean_over(fit_arm, full):+.3f} [{dd[50]:+.3f}, {dd[1949]:+.3f}]"
    else:
        diff = "ref"
    lam = lam_of(a)
    U.append(f"| {a} | {'' if lam is None else lam} | " + " | ".join(cells) + f" | {m:.3f} | {diff} |")
utxt = "\n".join(U)
print(utxt)
if len(sys.argv) > 2:
    open(sys.argv[2], "a").write(utxt + "\n")
