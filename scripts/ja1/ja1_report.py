"""ja1 report tables (hub, no torch): dev scores, split-half agreement, diagnostics, data accounting.
    python3 scripts/ja1/ja1_report.py 2b|9b"""
import glob
import json
import os
import re
import statistics as st
import sys

R = "/workspace/o-lens-improvement"
TAG = sys.argv[1]
O = f"{R}/runs/ja1/{TAG}"
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")


def summ(name):
    p = f"{R}/auto/runs/{name}/summary.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def lam_of(name):
    p = f"/root/ja1/{TAG}/eval_{name}.log"
    for f in (p, f"{O}/logs/eval_{name}.log"):
        if os.path.exists(f):
            m = re.findall(r"chosen lam ([0-9.]+)", open(f).read())
            if m:
                return m[-1]
    return ""


def row(label, name, lam=""):
    s = summ(name)
    if s is None:
        return f"| {label} | – | – | – | – | {' | '.join('–' for _ in TASKS)} |"
    pt = s["per_task_pass10"]
    return (f"| {label} | {s['pass10']:.4f} | {s['pass1']:.4f} | {s['pass10_unf']:.4f} | {lam} | "
            + " | ".join(f"{pt.get(t, float('nan')):.2f}" for t in TASKS) + " |")


print(f"## dev ({TAG})\n")
print("| lens | pass@10 | pass@1 | unf pass@10 | lam | " + " | ".join(TASKS) + " |")
print("|---|---|---|---|---|" + "---|" * len(TASKS))
print(row("J++ (released)", f"ja1_{TAG}_JPP"))
print(row("ridge ask-map + J++ (" + ("m2b_L" if TAG == "2b" else "fm27_F4") + ")", f"ja1_{TAG}_askmap",
          lam_of(f"ja1_{TAG}_askmap")))
for L in ["STD_n64", "W0_n32", "W012_n32", "A2_n248", "A5_n248", "A2_n40", "A5_n40"] + ["STD_n64", "A1_n64", "A2_n64", "A5_n64", "A3_n64", "A5u_n64", "A4_n64", "W012_n64",
          "A1_n256", "A2_n256", "A5_n256", "A3_n256", "A5u_n256", "W012_n256", "A1_n1024", "A2_n1024", "A5_n1024"]:
    for kind in ("alone", "add"):
        n = f"ja1_{TAG}_{L}_{kind}"
        if summ(n):
            print(row(f"{L} {kind}", n, lam_of(n) if kind == "add" else ""))

print(f"\n## split-half agreement ({TAG}): cos(flat) / top-10 overlap on 200 held-out activations, per layer\n")
for f in sorted(glob.glob(f"{O}/agree/*.json")):
    d = json.load(open(f))
    name = os.path.basename(f)[:-5]
    ls = sorted(d, key=int)
    print(f"{name:22s} " + "  ".join(f"L{l}:{d[l]['cos']:.3f}/{d[l]['top10']:.2f}" for l in ls)
          + f"   mean cos {st.mean(d[l]['cos'] for l in ls):.3f} top10 {st.mean(d[l]['top10'] for l in ls):.2f}")

print(f"\n## tr(J)/d of pooled lenses ({TAG})\n")
for f in sorted(glob.glob(f"{O}/lens/*_n*.json")):
    try:
        d = json.load(open(f))
    except Exception:  # noqa: BLE001
        continue
    if "tr_over_d" in d:
        print(f"{os.path.basename(f)[:-5]:22s} prompts {d['prompts']:5d} positions {d['positions']:6d}  "
              + " ".join(f"L{l}:{v:.3f}" for l, v in sorted(d["tr_over_d"].items(), key=lambda x: int(x[0]))))

print(f"\n## gradient mass ({TAG}): share on last passage token / first passage token / uniform 1/len; "
      "passage share of graph positions (passage + question)\n")
for r in ["A1", "A2", "A3", "A4", "W1", "W2"]:
    f = f"{O}/{r}/mass_summary.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"{r}: " + "  ".join(f"L{l}: last {v['last_share_of_passage']:.3f} first {v['first_share_of_passage']:.3f} "
                                   f"unif {v['uniform_share']:.3f} psg {v['passage_share_of_all_positions']:.2f}"
                                   for l, v in sorted(d.items(), key=lambda x: int(x[0]))))

print(f"\n## A5 expert position counts ({TAG})\n")
for f in sorted(glob.glob(f"{O}/lens/A5*_counts.json")):
    d = json.load(open(f))
    print(os.path.basename(f), "prompts", d["prompts"], {l: c for l, c in d["counts"].items()})

print(f"\n## data accounting ({TAG})\n")
for r in ["A1", "A2", "A3", "A4", "W1", "W2"]:
    f = f"{O}/{r}/prompts.json"
    if os.path.exists(f):
        P = json.load(open(f))
        ln = [len(p["pos"]) for p in P]
        info = json.load(open(f"{O}/{r}/fit_info.json")) if os.path.exists(f"{O}/{r}/fit_info.json") else {}
        for n in (64, 256, 1024):
            if n <= len(P):
                x = ln[:n]
                print(f"{r} n={n}: prompts {n}, source tokens {sum(x)} (passage len mean {st.mean(x):.1f}, "
                      f"min {min(x)}, max {max(x)}), targets {n}, pairs {sum(x)}, s/prompt {info.get('per_prompt', float('nan')):.1f}")
