"""W8 follow-up tables: base-fitted vs organism-fitted lenses (tops_<q>_all.pt), frozen keywords, organism / base control.
    python scripts/w8/compare_fit.py q1,q2   -> runs/w8/refit_compare.md
Tables per organism: (a) any-layer hit rate per lens x set x P1/P2; (b) per-layer, NEUTRAL / TRIGGER / PROBE at P1 and
P2; (c) count view: for each own keyword, number of prompts (of 40/40/10) with it in the top 10 at any layer and
position, organism vs base."""
import sys
from collections import Counter

sys.argv = [sys.argv[0], sys.argv[1]]
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
import analyze as A  # noqa: E402
from w8lib import LAYERS, OUT  # noqa: E402

QS = sys.argv[1].split(",")
LAB = {"J": "J (base-fit)", "JPP": "J++ (base-fit)", "A": "A (base-fit)", "JA": "J + A (base-fit)",
       "JPPA": "J++ + A (base-fit)", "rJ": "J (org-fit)", "rJPP": "J++ (org-fit)", "rA": "A (org-fit)",
       "rJA": "J + A (org-fit)", "rJPPA": "J++ + A (org-fit)"}
SETS = ("neutral", "trigger", "probe")
out = ["# W8 follow-up: lenses fitted on the organism vs on the base model\n",
       "Cells org / base %: share of prompts with any own-quirk keyword (runs/w8/keywords.json, frozen) in the top 10 "
       "at any layer and position of the set. org = organism activations, base = base model on the identical tokens; "
       "every lens (base-fitted or organism-fitted) is applied to both.\n"]
for q in QS:
    g, _ = A.load(q)
    import torch
    T = torch.load(f"{A.BIG}/tops_{q}_all.pt")
    kw = A.KWN[q]
    out.append(f"\n## {q}\n\n### Any layer\n")
    out.append("| lens | " + " | ".join(f"{s} P1 | {s} P2" for s in SETS) + " |")
    out.append("|---|" + "---|" * 6)
    H = {(w, l): A.hitmat(T, w, l, kw) for w in ("org", "base") for l in T["lenses"]}
    for l in T["lenses"]:
        c = []
        for s in SETS:
            its = [it for it in g if it["set"] == s]
            for pos in ("r1", "r2"):
                c.append(f"{A.fmt(A.rate(its, H['org', l], pos))} / {A.fmt(A.rate(its, H['base', l], pos))}")
        out.append(f"| {LAB[l]} | " + " | ".join(c) + " |")
    for s in SETS:
        for pos, pn in (("r1", "P1"), ("r2", "P2")):
            out.append(f"\n### Per layer, {s.upper()} {pn} (org/base %)\n")
            out.append("| lens | " + " | ".join(f"L{x}" for x in LAYERS) + " |")
            out.append("|---|" + "---|" * len(LAYERS))
            its = [it for it in g if it["set"] == s]
            for l in T["lenses"]:
                out.append(f"| {LAB[l]} | " + " | ".join(
                    f"{A.fmt(A.rate(its, H['org', l], pos, li))}/{A.fmt(A.rate(its, H['base', l], pos, li))}"
                    for li in range(len(LAYERS))) + " |")
    for kname in dict.fromkeys([q, "flattery"]):
      kw = A.KWN[kname]
      for s in SETS:
        for pos, pn in (("r1", "P1"), ("r2", "P2")):
            its = [it for it in g if it["set"] == s]
            out.append(f"\n### Count view ({kname} keywords), {s.upper()} {pn}: prompts (of {len(its)}) with the keyword "
                       f"in the top 10 at any layer/position, org/base\n")
            per = {}
            for l in T["lenses"]:
                for w in ("org", "base"):
                    c = Counter()
                    for it in its:
                        a, b = it[pos]
                        c.update({A.NF[x] for x in set(T["tops"][w][l][a:b, :, :10].reshape(-1).tolist())
                                  if x < len(A.NF) and A.NF[x] in kw})
                    per[w, l] = c
            words = sorted({k for v in per.values() for k in v}, key=lambda k: -sum(v[k] for v in per.values()))
            if not words:
                out.append("(no keyword in any top 10)")
                continue
            out.append("| keyword | " + " | ".join(LAB[l] for l in T["lenses"]) + " |")
            out.append("|---|" + "---|" * len(T["lenses"]))
            for k in words:
                out.append(f"| {k} | " + " | ".join(f"{per['org', l][k]}/{per['base', l][k]}" for l in T["lenses"]) + " |")
open(f"{OUT}/refit_compare.md", "w").write("\n".join(out) + "\n")
print("written", flush=True)
