"""q1 report (stdlib, hub side): D1 ceiling table next to the fitted lenses' top-10 hits (runs/qual/9b dumps).
    python3 scripts/q1/q1_report.py d1|d2|d3 [--tag 9b]
Groups: bench tasks (fit + dev rows), probe categories (NOT next-word, scorable probes only; is_next from the dumps)."""
import json
import sys
from collections import defaultdict

R = "/workspace/o-lens-improvement"
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "9b"
LENSES = {"J++": "jpp", "A": "a_lens", "J+A": "jpp_a_lens"}
K = 10


def hit(r):
    return r is not None and r <= K


def dumps():
    out = {}
    for k, s in LENSES.items():
        try:
            d = json.load(open(f"{R}/runs/qual/{TAG}/{s}.json", encoding="utf-8"))
        except FileNotFoundError:
            continue
        out[k] = {(it["set"], it["id"]): it for it in d["items"]}
    return out


def groups(rows, D):
    """-> list of (group name, [rows]) ; probe rows only if scorable and not next word (per the J++ dump)."""
    ref = next(iter(D.values()))
    g = defaultdict(list)
    for r in rows:
        if r["set"] == "probes":
            it = ref.get(("probes", r["id"]))
            if not it or not it.get("scorable") or it.get("is_next") is not False:
                continue
            g["P:" + r["task"]].append(r)
            g["P:ALL not-next"].append(r)
        else:
            g["B:" + r["task"]].append(r)
            g["B:ALL fit+dev"].append(r)
    return sorted(g.items(), key=lambda x: (x[0].startswith("P"), "ALL" in x[0], x[0]))


def lens_hit(D, name, r):
    it = D.get(name, {}).get((r["set"], r["id"]))
    return None if it is None else hit(it.get("best_rank"))


def d1():
    d = json.load(open(f"{R}/runs/q1/d1_{TAG}.json", encoding="utf-8"))
    D = dumps()
    rows = [r for r in d["rows"] if r.get("ranks") and r["ids"]]
    for r in rows:
        rk = r["ranks"]
        rk["q1_any"] = min([x for x in (rk[f"q1_topic_w{k}"] for k in range(3)) if x is not None], default=None)
        rk["q2_any"] = min([x for x in (rk[f"q2_unstated_w{k}"] for k in range(3)) if x is not None], default=None)
        r["q1_rate"] = sum(hit(rk[f"q1_topic_w{k}"]) for k in range(3)) / 3
    cols = ["q1_any", "q1_topic_mean", "q1_rate", "q2_any", "q2_unstated_mean"] + list(d["new_questions"]) + \
           [f"{s}{x}" for s in d["suffixes"] for x in ("_g0", "_g3")]
    short = {c: c.replace("q1_topic_mean", "q1_mst").replace("q2_unstated_mean", "q2_mst").replace("_referent", "ref")
             .replace("_intermediate", "int").replace("_canonical", "can").replace("_plan", "plan")
             .replace("_association", "asc").replace("_inotherwords", "iow").replace("_thatis", "ti")
             .replace("_namely", "nam").replace("_ie", "ie") for c in cols}
    print("D1 ask ceilings: % of rows whose concept is in the model's f4 top 10 answer (q*_any: any of 3 wordings; "
          "q*_mst: unembed of the mean unit final state over wordings = the A-lens target; q1_rate: mean over wordings; "
          "s*_g0: next token after the suffix, s*_g3: union over 3 greedy steps); lenses: best-layer top-10 hit.")
    hdr = ["group", "n"] + [short[c] for c in cols] + list(LENSES)
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    G = groups(rows, D)
    for name, rs in G:
        n = len(rs)
        vals = []
        for c in cols:
            if c == "q1_rate":
                vals.append(f"{100 * sum(r['q1_rate'] for r in rs) / n:.0f}")
            else:
                vals.append(f"{100 * sum(hit(r['ranks'][c]) for r in rs) / n:.0f}")
        for L in LENSES:
            h = [lens_hit(D, L, r) for r in rs]
            vals.append("-" if any(x is None for x in h) else f"{100 * sum(h) / n:.0f}")
        print(f"| {name} | {n} | " + " | ".join(vals) + " |")
    # A-lens hits / gains over J++ split by q1_any
    print("\nA-lens vs J++ split by whether the model's own topic answer (q1_any) contains the concept:")
    print("| group | q1 knows | n | J++ hits | A hits | A gains (A hit, J++ miss) | A losses |")
    print("|---|---|---|---|---|---|---|")
    for name, rs in G:
        if "ALL" not in name and not name.startswith("B:"):
            continue
        for kn in (True, False):
            sub = [r for r in rs if hit(r["ranks"]["q1_any"]) == kn]
            if not sub or "J++" not in D or "A" not in D:
                continue
            j = [lens_hit(D, "J++", r) for r in sub]
            a = [lens_hit(D, "A", r) for r in sub]
            print(f"| {name} | {'yes' if kn else 'no'} | {len(sub)} | {sum(j)} | {sum(a)} | "
                  f"{sum(x and not y for x, y in zip(a, j))} | {sum(y and not x for x, y in zip(a, j))} |")
    # examples
    if "--ex" in sys.argv:
        T = d["texts"]
        for name, rs in G:
            if not name.startswith("P:") or "ALL" in name:
                continue
            for r in rs[:2]:
                t = T[r["text"]]
                print(f"\n[{r['id']}] {r['text']!r} -> {r['concept']}")
                for c in ("q1_topic_w0", "q2_unstated_w0", "q3_referent", "q7_association", "s3_namely_g0"):
                    print(f"   {c} r={r['ranks'][c]}: {t['top'][c]}")
                print(f"   gen: {t['gen']}")



def d3():
    d = json.load(open(f"{R}/runs/q1/d3_{TAG}.json", encoding="utf-8"))
    L = d["layers"]
    print("| layer | cos W | top-50 input subspace mean cos² direct / via unembedding (random 0.012) | top-5 cosines (direct) | "
          "topic var in unstated top-50 PCs (own top-50) | unstated var in topic top-50 (own) | item cos centred / full | "
          "constant share of |out|² topic / unstated | top-10 overlap full / W-only |")
    print("|---|---|---|---|---|---|---|---|---|")
    for l in L:
        x = d["per_layer"][str(l)]
        v = x["var_shared"]
        print(f"| {l} | {x['cos_W']:.2f} | {x['angles_direct']['mean_cos2']:.2f} / {x['angles_logit']['mean_cos2']:.2f} | "
              f"{', '.join(f'{c:.2f}' for c in x['angles_direct']['top5'])} | {v['topic_in_unstated_top50']:.2f} ({v['topic_in_own_top50']:.2f}) | "
              f"{v['unstated_in_topic_top50']:.2f} ({v['unstated_in_own_top50']:.2f}) | {x['item_cos']:.2f} / {x['item_cos_full']:.2f} | "
              f"{x['const_share'][0]:.2f} / {x['const_share'][1]:.2f} | {x['top10_overlap']['full']:.1f} / {x['top10_overlap']['W_only']:.1f} |")
    # supply: rows where the A-lens (average) is a top-10 hit at some layer; at its best layer, which single map hits
    D = dumps()
    ref = D.get("J++", {})
    sup = defaultdict(lambda: defaultdict(int))
    for rid, r in d["rows"].items():
        if r["set"] == "probes":
            it = ref.get(("probes", rid))
            if not it or not it.get("scorable") or it.get("is_next") is not False:
                continue
            g = "probes not-next"
        else:
            g = "bench fit+dev"
        rk = r["ranks"]
        best = min(L, key=lambda l: (rk[str(l)]["alens"] if rk[str(l)]["alens"] is not None else 1e9))
        b = rk[str(best)]
        if not hit(b["alens"]):
            sup[g]["A miss"] += 1
            sup[g]["A miss, topic or unstated alone hits (any layer)"] += any(hit(rk[str(l)]["topic"]) or hit(rk[str(l)]["unstated"]) for l in L)
            continue
        t, u = hit(b["topic"]), hit(b["unstated"])
        sup[g]["A hit"] += 1
        sup[g]["both maps hit" if t and u else "topic map only" if t else "unstated map only" if u else "neither alone (sum only)"] += 1
    for g, c in sup.items():
        print(f"\n{g}: " + "; ".join(f"{k} {v}" for k, v in c.items()))
    ex = d["per_layer"][str(L[3])]["examples"]
    for k, v in list(ex.items())[:6]:
        print(f"\n...{k!r}")
        for m in ("topic", "unstated", "topic_W", "unstated_W"):
            print(f"   {m}: {v[m]}")



def d2():
    d = json.load(open(f"{R}/runs/q1/d2_{TAG}.json", encoding="utf-8"))
    d1 = {r["id"]: r for r in json.load(open(f"{R}/runs/q1/d1_{TAG}.json", encoding="utf-8"))["rows"]}
    D = dumps()
    L = d["layers"]
    rows = [r for r in d["rows"] if r.get("d2") and r["ids"]]
    qk = sorted({k.split("|")[1] for k in rows[0]["d2"]})
    tasks = {"topic": [k for k in qk if k.startswith("q1_topic")]}
    for k in qk:
        if not k.startswith("q1_topic"):
            tasks[k] = [k]
    A = D["A"]

    def abest(r):
        it = A.get((r["set"], r["id"]))
        return L.index(it["best_layer"]) if it and it.get("best_layer") in L else L.index(16)

    def anyhit(r, ph, ks, kind, j=None):
        out = False
        for k in ks:
            x = r["d2"][f"{ph}|{k}"][kind]
            if kind == "none":
                out |= hit(x)
            elif j is None:
                out |= any(hit(v) for v in x)
            else:
                out |= hit(x[j])
        return out

    def ask(r, ks):
        return any(hit(d1[r["id"]]["ranks"][k]) for k in ks)

    G = groups(rows, D)
    for tname, ks in tasks.items():
        print(f"\n### D2 {tname} ({'any of ' + str(len(ks)) + ' wordings' if len(ks) > 1 else ks[0]}): % rows with the concept in the answer top 10")
        print("| group | n | ask (D1) | x: none | x: own@A-best | x: own any layer | x: other@A-best | bland: none | bland: own@A-best | bland: own any | bland: other@A-best | A-lens fitted |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rs in G:
            n = len(rs)
            v = [sum(ask(r, ks) for r in rs)]
            for ph in ("ph_x", "ph_bland"):
                v += [sum(anyhit(r, ph, ks, "none") for r in rs), sum(anyhit(r, ph, ks, "own", abest(r)) for r in rs),
                      sum(anyhit(r, ph, ks, "own") for r in rs), sum(anyhit(r, ph, ks, "other", abest(r)) for r in rs)]
            v.append(sum(bool(lens_hit(D, "A", r)) for r in rs))
            print(f"| {name} | {n} | " + " | ".join(f"{100 * x / n:.0f}" for x in v) + " |")
        for name, rs in G:
            if "ALL" not in name:
                continue
            for ph in ("ph_x", "ph_bland"):
                print(f"   {name} {ph} own patch by layer: " + ", ".join(
                    f"L{l} {100 * sum(anyhit(r, ph, ks, 'own', j) for r in rs) / len(rs):.0f}" for j, l in enumerate(L)) +
                      " | other-item patch: " + ", ".join(
                    f"L{l} {100 * sum(anyhit(r, ph, ks, 'other', j) for r in rs) / len(rs):.0f}" for j, l in enumerate(L)))
        # overlap: own-patch hits vs fitted A-lens hits
        for name, rs in G:
            if "ALL" not in name:
                continue
            p = [any(anyhit(r, ph, ks, "own", abest(r)) for ph in ("ph_x", "ph_bland")) for r in rs]
            a = [bool(lens_hit(D, "A", r)) for r in rs]
            print(f"   {name}: patched (either placeholder, A-best layer) & A-lens {sum(x and y for x, y in zip(p, a))}, "
                  f"patched only {sum(x and not y for x, y in zip(p, a))}, A-lens only {sum(y and not x for x, y in zip(p, a))}, n {len(rs)}")
    print("\n### named-entity control (A-lens vs J++, best layer over readout layers)")
    nh = {"A-lens": 0, "J++": 0}
    nn = 0
    for r in d["named"]:
        if r["next_rank"] is not None and r["next_rank"] <= 10:
            print(f"   (next word, excluded) {r['id']} {r['text']!r} -> {r['concept']}: next {r['next'][:6]}")
            continue
        nn += 1
        line = []
        for m in ("J++", "A-lens"):
            b = min(r["lens"][m], key=lambda e: e["r"] if e["r"] is not None else 1e9)
            j = r["lens"][m].index(b)
            nh[m] += hit(b["r"])
            line.append(f"{m} {b['r']}@L{L[j]} {b['t']}")
        print(f"   {r['id']} {r['text']!r} -> {r['concept']}: " + " || ".join(line))
    print(f"   hits (not-next prompts, n {nn}): " + ", ".join(f"{k} {v}" for k, v in nh.items()))


if __name__ == "__main__":
    {"d1": d1, "d3": d3, "d2": d2}[sys.argv[1]]()
