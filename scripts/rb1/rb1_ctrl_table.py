"""rb1 control table: dev scores of control / truncated-map lenses (+ chosen lam), per-layer diagnostics, items log.
Usage: python rb1_ctrl_table.py <tag>  (pod)"""
import json
import re
import sys

R = "/workspace/o-lens-improvement"
TAG = sys.argv[1]


def row(v):
    n = f"rb1_{v}_{TAG}"
    try:
        s = json.load(open(f"{R}/auto/runs/{n}/summary.json"))
    except FileNotFoundError:
        return f"{v:16s} -"
    try:
        lam = re.search(r"chosen lam ([0-9.]+)", open(f"/root/rb1/eval_{n}.log").read()).group(1)
    except Exception:
        lam = "?"
    return f"{v:16s} lam {lam:4s} p10 {100 * s['pass10']:.1f} p1 {100 * s['pass1']:.1f} unf-p10 {100 * s['pass10_unf']:.1f}"


print(f"== {TAG} references")
for v in ("jppu", "Lstored", "r300_0", "r3000_0", "rall"):
    print("  " + row(v))
print("== controls (lam chosen on FIT / at the reference lens's lam: all -> rall = L, 300 -> r300_0, 3000 -> r3000_0)")
for v in ("shufall", "priorall", "shuf300", "prior300", "shuf3000", "prior3000"):
    print("  " + row(v) + "   |   at L: " + row(v + "_atL")[17:])
print("== map replaced by its top-k singular directions (same mx, b; lam chosen on FIT)")
for v in ("r300_0", "r3000_0", "rall"):
    for k in (1, 4, 16, 64, 256):
        print("  " + row(f"{v}_k{k}"))
C = json.load(open(f"{R}/.hf27/rb1/ctrl_{TAG}.json"))
print("== per-layer: chosen c / eff. rank of W (exp-entropy) / eff. rank of centred output on test / b energy share / cos(Eh+b, b)")
for v, d in C["diag"].items():
    print(f"  {v:8s} " + "  ".join(f"L{l} c={x['c']:g} erW={x['erank_W']:.0f} erO={x['erank_out']:.0f} bE={x['b_energy']:.2f} cos={x['cos_b']:.2f}"
                                     for l, x in d.items()))
    print(f"  {'':8s} mean " + " ".join(f"{k} {x:.3f}" for k, x in C["mean"][v].items()))
try:
    print(open(f"/root/rb1/items_{TAG}.log").read())
except FileNotFoundError:
    pass
