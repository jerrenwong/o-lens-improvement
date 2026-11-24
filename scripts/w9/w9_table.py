"""w9 analysis (hub python3, no torch): grid table, bootstraps, fit-vs-benchmark, per-task scores.
python3 scripts/w9/w9_table.py [tags...]  -> prints markdown, writes runs/w9/table.md"""
import json
import re
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w7"))
import w7_boot as B  # noqa: E402

TAGS = sys.argv[1:] or ["9b", "2b"]
ARCH = ["lin", "mlp512", "mlp2048"]
LOSS = ["l2", "ce"]
NS = ["28k", "98k"]
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)


def summ(name):
    p = R / f"auto/runs/{name}/summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def lam(name):
    for p in (R / f"runs/w9/logs/{name}.log", R / f"runs/w7/logs/{name}.log"):
        if p.exists():
            m = re.findall(r"chosen lam ([0-9.]+)", p.read_text())
            return m[-1] if m else "-"
    return "-"


def cell(name):
    s = summ(name)
    return "n/a" if s is None else f"{s['pass10']:.3f} / {s['pass1']:.3f}"


def fitloss(tag, a, lo, n, suf=""):
    vals, trs = [], []
    for q in ("topic", "spec"):
        p = R / f"runs/w9/fits/E_{tag}_{q}_{a}_{lo}_{n}{suf}.json"
        if not p.exists():
            return None
        L = json.loads(p.read_text())["layers"]
        vals += [v["val"] for v in L.values()]; trs += [v["train"] for v in L.values()]
    steps = None
    if a != "lin" or lo == "ce":
        steps = [v["step"] for q in ("topic", "spec")
                 for v in json.loads((R / f"runs/w9/fits/E_{tag}_{q}_{a}_{lo}_{n}{suf}.json").read_text())["layers"].values()]
    return sum(trs) / len(trs), sum(vals) / len(vals), steps


for tag in TAGS:
    P(f"\n## {tag}: dev top-10 / top-1 (lam) — A-lens alone | J++ + A-lens, filters f4 and f1\n")
    P("| map | n | f4 alone | f4 +J++ (lam) | f1 alone | f1 +J++ (lam) | train / val loss (mean over layers, both Qs) | early-stop steps |")
    P("|---|---|---|---|---|---|---|---|")
    P(f"| J++ alone | - | - | {cell(f'w7_{tag}_JPP0_f4')} | - | {cell(f'w7_{tag}_JPP0_f1')} | | |")
    P(f"| ridge A-lens (current, w7 ref) | 28k | {cell(f'w7_{tag}_ridge_none_f4')} | {cell(f'w7_{tag}_ridge_JPP_f4')} ({lam(f'w7_{tag}_ridge_JPP_f4')}) | "
      f"{cell(f'w7_{tag}_ridge_none_f1')} | {cell(f'w7_{tag}_ridge_JPP_f1')} ({lam(f'w7_{tag}_ridge_JPP_f1')}) | | |")
    for lo in LOSS:
        for n in NS:
            for a in ARCH:
                s = f"w9_{tag}_{a}_{lo}_{n}"
                fl = fitloss(tag, a, lo, n)
                fls = "" if fl is None else f"{fl[0]:.4f} / {fl[1]:.4f}"
                st = "" if fl is None or fl[2] is None else f"{sum(x == 0 for x in fl[2])}/14 at 0; median {sorted(fl[2])[7]}"
                P(f"| {a} {lo} | {n} | {cell(s + '_none_f4')} | {cell(s + '_JPP_f4')} ({lam(s + '_JPP_f4')}) | {cell(s + '_none_f1')} | "
                  f"{cell(s + '_JPP_f1')} ({lam(s + '_JPP_f1')}) | {fls} | {st} |")
    for sh in ("_shuf",):
        for a in ARCH:
            for lo in LOSS:
                for n in NS:
                    s = f"w9_{tag}_{a}_{lo}_{n}{sh}"
                    if summ(s + "_JPP_f4") or summ(s + "_none_f4"):
                        P(f"| {a} {lo} SHUFFLED | {n} | {cell(s + '_none_f4')} | {cell(s + '_JPP_f4')} ({lam(s + '_JPP_f4')}) | "
                          f"{cell(s + '_none_f1')} | {cell(s + '_JPP_f1')} ({lam(s + '_JPP_f1')}) | | |")

    P(f"\n### {tag}: paired bootstrap (2000 draws, dev items resampled within task), filter f4: diff top-10 [95% CI] | top-1 [95% CI]\n")
    P("| comparison | row | top-10 | top-1 |"); P("|---|---|---|---|")

    def bline(a_, b_, lab, row):
        if summ(a_) and summ(b_):
            (d10, l10, h10), (d1, l1, h1) = B.boot(a_, b_)
            P(f"| {lab} | {row} | {d10:+.3f} [{l10:+.3f}, {h10:+.3f}] | {d1:+.3f} [{l1:+.3f}, {h1:+.3f}] |")
    for row in ("none", "JPP"):
        for lo in LOSS:
            for n in NS:
                for a in ("mlp512", "mlp2048"):
                    bline(f"w9_{tag}_lin_{lo}_{n}_{row}_f4", f"w9_{tag}_{a}_{lo}_{n}_{row}_f4", f"{a} - lin ({lo}, {n})", row)
        for lo in LOSS:
            for a in ARCH:
                bline(f"w9_{tag}_{a}_{lo}_28k_{row}_f4", f"w9_{tag}_{a}_{lo}_98k_{row}_f4", f"98k - 28k ({a}, {lo})", row)
    P(f"\n### {tag}: per-task dev top-10 (f4)\n")
    tasks = ["association", "multihop", "multilingual", "poetry", "typo"]
    P("| run | " + " | ".join(tasks) + " |"); P("|---" * (len(tasks) + 1) + "|")
    for nm in [f"w7_{tag}_JPP0_f4", f"w7_{tag}_ridge_none_f4", f"w7_{tag}_ridge_JPP_f4"] + \
            [f"w9_{tag}_{a}_{lo}_{n}_{row}_f4" for row in ("none", "JPP") for lo in LOSS for n in NS for a in ARCH]:
        s = summ(nm)
        if s:
            P(f"| {nm} | " + " | ".join(f"{s['per_task_pass10'].get(t, float('nan')):.3f}" for t in tasks) + " |")
(R / "runs/w9/table.md").write_text("\n".join(out) + "\n")
