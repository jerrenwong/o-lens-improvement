"""ej1: per-category hits and near_wrong rates on probes_v2 dumps (stdlib).
    python3 scripts/ej1/v2_stats.py probes/probes_v2.json runs/qual/9b_v2/a.json runs/qual/9b_v2/b.json ... [--k 10]
hit = best accepted rank <= k over layers (qualfmt best_rank). near_wrong: for a MISSED probe that has a near_wrong list,
does a near_wrong word appear in the lens's top-k (a) at any readout layer, (b) at the layer of its best accepted rank.
Words match the stripped, lower-cased token, or a token that is a prefix (>= 3 chars) of the word."""
import json
import sys
from collections import defaultdict


def norm(s):
    return s.strip().lower()


def match(tok, words):
    t = norm(tok)
    if not t:
        return False
    return any(t == w or (len(t) >= 3 and w.startswith(t)) for w in words)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 10
    probes = json.load(open(args[0]))
    probes = probes.get("probes", probes) if isinstance(probes, dict) else probes
    P = {p["id"]: p for p in probes}
    dumps = [json.load(open(a)) for a in args[1:]]
    names = [d["meta"]["lens"] for d in dumps]
    cats = sorted({p.get("category") or p.get("task") for p in probes})
    tab = {n: defaultdict(lambda: [0, 0]) for n in names}
    nw = {n: [0, 0, 0] for n in names}  # missed-with-near_wrong, any-layer, best-layer
    for n, d in zip(names, dumps):
        for it in d["items"]:
            if it.get("set") != "probes" or it["id"] not in P or not it.get("scorable", True):
                continue
            p = P[it["id"]]
            c = p.get("category") or p.get("task")
            r = it.get("best_rank")
            hit = r is not None and r <= k
            tab[n][c][0] += hit
            tab[n][c][1] += 1
            tab[n]["ALL"][0] += hit
            tab[n]["ALL"][1] += 1
            words = [norm(w) for w in (p.get("near_wrong") or [])]
            if words and not hit:
                nw[n][0] += 1
                tops = [L["t"][:k] for L in it["layers"]]
                nw[n][1] += any(match(t, words) for T in tops for t in T)
                bl = it.get("best_layer")
                lay = d["meta"]["layers"]
                T = tops[lay.index(bl)] if bl in lay else []
                nw[n][2] += any(match(t, words) for t in T)
    print("| category | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for c in ["ALL"] + cats:
        nn = tab[names[0]][c][1]
        print(f"| {c} | {nn} | " + " | ".join(str(tab[x][c][0]) for x in names) + " |")
    print("\nnear_wrong while missing (missed probes with a near_wrong list: share with a near_wrong word in top-%d)" % k)
    print("| lens | missed w/ list | any layer | best layer |")
    print("|---|---|---|---|")
    for x in names:
        m, a, b = nw[x]
        print(f"| {x} | {m} | {a} ({100 * a / max(1, m):.0f}%) | {b} ({100 * b / max(1, m):.0f}%) |")


if __name__ == "__main__":
    main()
