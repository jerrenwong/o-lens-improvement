"""w1: qualitative look at fit-split top-10 lists under f1 ("f"), f2, f3 (runs/fitA_<tag>_<mode>.json from fitA_dump.py)."""
import collections, json, random, sys
R = "/workspace/o-lens-improvement/runs"
for tag in ("9b", "2b"):
    for mode in ("JPP0", "none", "JPP"):
        try:
            d = json.load(open(f"{R}/fitA_{tag}_{mode}.json"))
        except FileNotFoundError:
            continue
        print(f"=== {tag} {mode} lam {d.get('lam')} {d.get('filters')}")
        for lv in ("f", "f2", "f3", "u"):
            c1, ct, n, hit = collections.Counter(), collections.Counter(), 0, 0
            for it in d["items"].values():
                for L in d["layers"]:
                    v = it[str(L)] if lv == "f" else it[str(L)].get(lv)
                    if v is None: continue
                    n += 1; c1[v["top"][0]] += 1; ct.update(v["top"]); hit += any(v["hit"])
            if n:
                print(f"  {lv:2s} rows {n} rows-with-hit {hit} | top1: " + ", ".join(f"{t!r}:{c}" for t, c in c1.most_common(6)))
                print(f"     in top10: " + ", ".join(f"{t!r}:{c}" for t, c in ct.most_common(12)))
        random.seed(0)
        for k in random.sample(sorted(d["items"]), 3):
            L = str(d["layers"][len(d["layers"]) // 2])
            it = d["items"][k]
            print(f"  -- {k} layer {L}")
            for lv in ("f", "f2", "f3"):
                v = it[L] if lv == "f" else it[L].get(lv)
                if v: print(f"     {lv:2s} rank {v['rank']} {[t for t in v['top']]}")
