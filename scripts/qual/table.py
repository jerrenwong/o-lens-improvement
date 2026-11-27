"""N-lens overview of qual dumps of one model: hits per set and per task/category for each lens, and the items where
the lenses disagree (some hit, some miss), with each lens's best rank and layer.

    python3 scripts/qual/table.py runs/qual/9b/jpp.json runs/qual/9b/a_lens.json runs/qual/9b/jpp_a_lens.json
            [--set all|probes|bench-fit|bench-dev] [--layer best|<n>] [--k 10] [--max 60]

Columns follow the argument order. Only items present in every dump are counted (ids matched).
Hit = best accepted rank <= k at the chosen layer; controls and unscorable probes are not counted.
Groups: set, task/category, difficulty, next word / not next word (when the dumps know is_next).
Disagreement rows: "id | concept | rank@layer per lens", sorted by the spread of log ranks.
Stdlib only (Python 3.8).
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualfmt as Q  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--set", default="all", choices=["all", "probes", "bench-fit", "bench-dev"])
    ap.add_argument("--layer", default="best")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max", type=int, default=60, help="max disagreement rows printed")
    a = ap.parse_args(argv)
    D = [Q.load(p) for p in a.dumps]
    models = {d["meta"]["model"] for d in D}
    if len(models) > 1:
        sys.exit(f"dumps are from different models: {sorted(models)}")
    names = [d["meta"]["lens"] for d in D]
    maps = [{it["id"]: it for it in Q.select(d, a.set)} for d in D]
    ids = [i for i in maps[0] if all(i in m for m in maps[1:])]
    rows = []
    for i in ids:
        its = [m[i] for m in maps]
        if not Q.scorable(its[0]):
            continue
        reads = [Q.view(it, d["meta"], a.layer) for it, d in zip(its, D)]
        if any(e.get("r") is None for _, e in reads):
            continue
        rows.append(dict(id=i, it=its[0], ranks=[e.get("r") for _, e in reads], lays=[L for L, _ in reads]))
    print(f"# {D[0]['meta']['model']} | set {a.set} | layer {a.layer} | k {a.k} | {len(rows)} items")
    print("lenses: " + " | ".join(f"{c}={n}" for c, n in zip("ABCDEFGHIJ", names)))

    def line(label, sel):
        hits = [sum(Q.hit(r["ranks"][j], a.k) for r in sel) for j in range(len(D))]
        return f"| {label} | {len(sel)} | " + " | ".join(str(h) for h in hits) + " |"

    cols = " | ".join("ABCDEFGHIJ"[j] for j in range(len(D)))
    print(f"\n| group | n | {cols} |\n|---|---|" + "---|" * len(D))
    print(line("ALL", rows))
    for key in ("set", "task", "difficulty", "is_next"):
        g = {}
        for r in rows:
            g.setdefault((Q.next_label(r["it"]) or "unknown") if key == "is_next" else (r["it"].get(key) or "-"), []).append(r)
        if key in ("difficulty", "is_next") and set(g) <= {"-", "unknown"}:
            continue
        if key == "set" and len(g) == 1:
            continue
        for v, sel in sorted(g.items()):
            print(line(f"{key}: {v}", sel))
    dis = [r for r in rows if len({Q.hit(x, a.k) for x in r["ranks"]}) > 1]

    def spread(r):
        lg = [math.log10(x) for x in r["ranks"] if x is not None]
        return max(lg) - min(lg) if lg else 0

    dis.sort(key=spread, reverse=True)
    print(f"\n## disagree ({len(dis)}; showing {min(len(dis), a.max)}): rank@layer per lens, * = hit")
    for r in dis[: a.max]:
        cells = [f"{'*' if Q.hit(x, a.k) else ' '}{Q.fmt_rank(x)}@L{L}" for x, L in zip(r["ranks"], r["lays"])]
        print(f"- {r['id']} | {r['it'].get('concept') or '/'.join(r['it']['accepted'])} | " + " | ".join(
            f"{c}:{s}" for c, s in zip("ABCDEFGHIJ", cells)))


if __name__ == "__main__":
    main()
