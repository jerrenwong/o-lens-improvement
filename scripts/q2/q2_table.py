"""q2: one table over many qual dumps of one model (stdlib; reads runs/qual/<tag>/<slug>.json and the dev scores the
dump's evaluator run wrote to auto/runs/qual_<tag>_<slug>/summary.json).
    python3 scripts/q2/q2_table.py <tag> <slug> [<slug> ...] [--cats]
Columns: dev f4 pass@10 / pass@1 and per task (typo, poetry, ...); not-next-word probe hits (k 10, best layer);
next-word probe hits; bench-dev hits; recurring-token statistics on the not-next probes (top-10 lists at the best
layer): share of top-10 slots taken by the 5 most recurring tokens, and the 5 tokens with the number of probes holding
them. --cats: per-category not-next probe hits."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/qual")
import qualfmt as Q  # noqa: E402

R = "/workspace/o-lens-improvement"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cats = "--cats" in sys.argv
    tag, slugs = args[0], args[1:]
    rows, catrows, allcats = [], [], set()
    for s in slugs:
        d = Q.load(f"{R}/runs/qual/{tag}/{s}.json")
        sm = f"{R}/auto/runs/qual_{tag}_{s}/summary.json"
        S = json.load(open(sm)) if os.path.exists(sm) else {}
        pt = S.get("per_task_pass10", {})
        P = [it for it in Q.select(d, "probes") if Q.scorable(it) and it.get("accepted")]
        nn = [it for it in P if it.get("is_next") is False]
        nx = [it for it in P if it.get("is_next") is True]
        bd = [it for it in Q.select(d, "bench-dev") if Q.scorable(it)]

        def hits(items):
            return sum(Q.hit(Q.view(it, d["meta"], "best")[1].get("r"), 10) for it in items)
        rec, slots = Counter(), 0
        for it in nn:
            e = Q.view(it, d["meta"], "best")[1]
            toks = {Q.norm_tok(t) for t in e["t"][:10]}
            rec.update(toks); slots += 10
        top5 = rec.most_common(5)
        share = sum(c for _, c in top5) / max(slots, 1)
        lam = ""
        rows.append([d["meta"]["lens"], f"{S.get('pass10', float('nan')):.3f}", f"{S.get('pass1', float('nan')):.3f}",
                     *(f"{pt.get(t, float('nan')):.2f}" for t in ("association", "multihop", "multilingual", "poetry", "typo")),
                     f"{hits(nn)}/{len(nn)}", f"{hits(nx)}/{len(nx)}", f"{hits(bd)}/{len(bd)}", f"{100 * share:.0f}%",
                     ", ".join(f"{w} {c}" for w, c in top5)])
        if cats:
            cc = {}
            for it in nn:
                c = it.get("category") or it.get("task") or "-"
                allcats.add(c)
                h, n = cc.get(c, (0, 0))
                cc[c] = (h + Q.hit(Q.view(it, d["meta"], "best")[1].get("r"), 10), n + 1)
            catrows.append((d["meta"]["lens"], cc))
        del lam
    hdr = ["lens", "dev p@10", "dev p@1", "assoc", "multihop", "multiling", "poetry", "typo", "probes not-next",
           "probes next", "bench-dev", "top-5 recurring share (not-next)", "recurring (probes holding it)"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    if cats:
        cl = sorted(allcats)
        print()
        print("| lens | " + " | ".join(c.replace("-", " ") for c in cl) + " |")
        print("|" + "---|" * (len(cl) + 1))
        for name, cc in catrows:
            print(f"| {name} | " + " | ".join(f"{cc.get(c, (0, 0))[0]}/{cc.get(c, (0, 0))[1]}" for c in cl) + " |")


if __name__ == "__main__":
    main()
