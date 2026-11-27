"""Side-by-side WorkspaceBench fast results for several lenses, plus the units where two lenses differ.

Usage (pod, benchmark env, so the family-specific unit rules of wsbench.fast.units are used):
  /root/wsb_env/bin/python report.py jlens=runs/wsb/openai_gpt-6-luna/jlens jpp=runs/wsb/openai_gpt-6-luna/jpp \
        [--out runs/wsb/openai_gpt-6-luna/compare_jlens_jpp] [--boot 2000]
  (python3 without wsbench also works: values and CIs from results.json; units from rows' `pass` only, approximate)
Writes <out>.md and <out>_diff_units.jsonl. Prints the table.
Table: per family, each lens's fast value [the tool's 95% CI: unit bootstrap, extras.fast_ci] (n scored units,
  k undecided), then the paired difference of the LAST lens minus the FIRST on the units both scored, with a
  paired unit bootstrap CI (2,000 resamples, seed 0). Macro = unweighted mean over the families every lens has;
  its paired CI resamples units within each family. NOT a number of record: an override judge's rows are never
  `pinned`/`complete`, so `wsbench report` leaves its own macro empty for them; this macro is ours.
Diff file: one JSON line per (family, unit) whose outcome differs between any two lenses:
  {"family", "unit", "outcomes": {lens: value}, "results": {lens: results.json path},
   "readouts": {lens: readouts file from the run's config, if recorded}} - input for a later qualitative compare.
"""
import argparse
import json
import os
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="name=dir (a judged out dir with <family>/results.json)")
ap.add_argument("--out", default="")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
runs = [r.split("=", 1) for r in a.runs]

try:
    from wsbench import fast as fast_mod
    from wsbench.results import read_results
    HAVE_WSB = True
except Exception:
    HAVE_WSB = False
    print("wsbench not importable: approximate units from rows' `pass`", file=sys.stderr)


def load(d):
    out = {}
    for fam in sorted(os.listdir(d)):
        p = os.path.join(d, fam, "results.json")
        if not os.path.exists(p):
            continue
        raw = json.load(open(p))
        num = raw.get("numbers", {})
        ex = num.get("extras", {}) or {}
        units = {}
        if HAVE_WSB:
            try:
                res = read_results(os.path.join(d, fam))
                groups, sign = fast_mod.units(fam, res)
                if list(sign) == ["all"]:
                    units = {str(k): float(v) for k, v in groups["all"].items()}
            except Exception as e:
                print(f"{d}/{fam}: units failed ({type(e).__name__}: {e})", file=sys.stderr)
        if not units:
            for r in raw.get("rows", []):
                if isinstance(r.get("pass"), bool):
                    units[str(r["id"])] = float(r["pass"])
        out[fam] = dict(value=num.get("value"), ci=num.get("ci95"), n=raw.get("n_items"),
                        undecided=ex.get("n_items_undecided"), complete=raw.get("complete"),
                        judge=(raw.get("config") or {}).get("judge_model"), units=units, path=p,
                        readouts=(raw.get("config") or {}).get("readouts"))
    return out


R = {name: load(d) for name, d in runs}
names = [n for n, _ in runs]
fams = sorted(set.union(*(set(v) for v in R.values())))
common = [f for f in fams if all(f in R[n] and R[n][f]["value"] is not None for n in names)]
A, B = names[0], names[-1]
rng = random.Random(0)


def fmt(x):
    if x is None:
        return "-"
    v = x["value"]
    ci = x["ci"] or [None, None]
    s = f"{v:.3f}" if v is not None else "-"
    if ci[0] is not None:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    s += f" ({x['n']}" + (f", {x['undecided']} und." if x.get("undecided") else "") + ")"
    return s


def paired(fam):
    ua, ub = R[A][fam]["units"], R[B][fam]["units"]
    shared = sorted(set(ua) & set(ub))
    return [(ua[u], ub[u]) for u in shared]


draws = {}
lines = [f"Lenses: {', '.join(f'{n} = {d}' for n, d in runs)}", "",
         "| family | " + " | ".join(names) + f" | {B} - {A} (paired units) [95% CI] |",
         "|---|" + "---|" * (len(names) + 1)]
for f in fams:
    cells = [fmt(R[n].get(f)) for n in names]
    d = "-"
    if f in common and len(names) > 1:
        pr = paired(f)
        if pr:
            diff = sum(b - x for x, b in pr) / len(pr)
            bs = sorted(sum(b - x for x, b in (rng.choice(pr) for _ in pr)) / len(pr) for _ in range(a.boot))
            draws[f] = pr
            d = f"{diff:+.3f} [{bs[int(0.025 * a.boot)]:+.3f}, {bs[int(0.975 * a.boot) - 1]:+.3f}] (n={len(pr)})"
    lines.append(f"| {f} | " + " | ".join(cells) + f" | {d} |")
macro = {n: sum(R[n][f]["value"] for f in common) / len(common) for n in names} if common else {}
md = "-"
if len(names) > 1 and draws:
    fs = sorted(draws)
    point = sum(sum(b - x for x, b in draws[f]) / len(draws[f]) for f in fs) / len(fs)
    bs = sorted(sum(sum(b - x for x, b in (rng.choice(draws[f]) for _ in draws[f])) / len(draws[f]) for f in fs) / len(fs)
                for _ in range(a.boot))
    md = f"{point:+.3f} [{bs[int(0.025 * a.boot)]:+.3f}, {bs[int(0.975 * a.boot) - 1]:+.3f}] ({len(fs)} families)"
lines.append(f"| **macro over {len(common)} common families** | " + " | ".join(f"{macro[n]:.3f}" if macro else "-" for n in names)
             + f" | {md} |")
judges = sorted({str(R[n][f]["judge"]) for n in names for f in R[n]})
lines += ["", f"Judge(s): {', '.join(judges)}. Override-judge rows: not numbers of record (docs/judges.md)."]
missing = {n: [f for f in fams if f not in R[n]] for n in names}
if any(missing.values()):
    lines.append("Missing families: " + "; ".join(f"{n}: {', '.join(v)}" for n, v in missing.items() if v))
txt = "\n".join(lines)
print(txt)

diffs = []
for f in fams:
    ids = sorted(set.union(*(set(R[n][f]["units"]) for n in names if f in R[n])))
    for u in ids:
        oc = {n: R[n][f]["units"].get(u) for n in names if f in R[n]}
        if len({v for v in oc.values() if v is not None}) > 1:
            diffs.append(dict(family=f, unit=u, outcomes=oc,
                              results={n: R[n][f]["path"] for n in names if f in R[n]},
                              readouts={n: R[n][f]["readouts"] for n in names if f in R[n]}))
if a.out:
    open(a.out + ".md", "w").write(txt + "\n")
    with open(a.out + "_diff_units.jsonl", "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in diffs))
print(f"\n{len(diffs)} units differ between lenses" + (f" -> {a.out}_diff_units.jsonl" if a.out else ""))
