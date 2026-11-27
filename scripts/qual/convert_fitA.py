"""Convert the existing fit-split A-lens dumps (runs/fitA_<tag>_<mode>.json, auto/candidates/fitA_dump.py) to qual dumps.

    python3 scripts/qual/convert_fitA.py            # all of 9b, 2b x none / JPP / JPP0 that exist
    python3 scripts/qual/convert_fitA.py --tags 9b

mode -> lens: JPP0 = "J++", none = "A-lens", JPP = "J++ + A-lens". Filter f4 (word groups), top-10 only (that is all
fitA holds), set "bench-fit". Prompts, concepts and accepted token forms come from the project's viewer data
(/workspace/fleet/projects/o-lens/site/data/jpp_fit_<tag>.json, same items). Items with several concepts (multilingual)
become one row per concept: id "<task>|<name>#<concept>". Writes runs/qual/<tag>/<slug>.json.
Stdlib only (Python 3.8).
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualfmt as Q  # noqa: E402

R = "/workspace/o-lens-improvement"
SITE = "/workspace/fleet/projects/o-lens/site/data"
MODELS = {"9b": "Qwen/Qwen3.5-9B", "2b": "Qwen/Qwen3.5-2B"}
MODES = {"JPP0": ("J++", "unit(J++ h): the released J++ lens alone (alens.py ALENS_BASE=JPP0)"),
         "none": ("A-lens", "a(h): average of the two ridge ask-maps, no Jacobian (alens.py ALENS_BASE=none)"),
         "JPP": ("J++ + A-lens", "unit(J++ h) + lam * a(h), lam chosen on the fit split (alens.py ALENS_BASE=JPP)")}


def clean(s):
    return s.replace("�", "▯")


def convert(tag, mode):
    src = f"{R}/runs/fitA_{tag}_{mode}.json"
    if not os.path.exists(src):
        print(f"skip {src} (missing)")
        return None
    A = json.load(open(src, encoding="utf-8"))
    V = json.load(open(f"{SITE}/jpp_fit_{tag}.json", encoding="utf-8"))
    voc = V["vocab"]
    layers = [int(x) for x in A["layers"]]
    assert layers == [int(x) for x in V["layers"]], (layers, V["layers"])
    name, how = MODES[mode]
    lam = A.get("lam")
    if mode == "JPP" and lam is not None:
        how += f"; lam = {lam}"
    items = []
    for it in V["items"]:
        key = f"{it['e']}|{it['n']}"
        g = A["items"][key]
        text = clean("".join(voc[t] for t in it["toks"][: it["pos"] + 1]))
        w = it.get("w") or {}
        model = {}
        if w.get("nx"):
            model["next"] = [clean(t) for t in w["nx"]]
        if w.get("a"):
            model["asked"] = clean(w["a"])
        multi = len(it["inter"]) > 1
        for ci, concept in enumerate(it["inter"]):
            toks = [voc[c] for c in it["cand"][ci]]
            words = []
            for t in toks:
                s = t.strip()
                if s and s not in words:
                    words.append(s)
            entries = []
            for L in layers:
                f4 = g[str(L)]["f4"]
                top = [clean(t) for t in f4["top"]]
                ranks = f4["rank"]
                # hit flags: the word group of the token holds a candidate of ANY concept of the item; keep only the
                # ones that match this concept's accepted words (by normal form) when the item has several concepts
                h = [i for i, x in enumerate(f4["hit"]) if x]
                if multi:
                    forms = {Q.normal_form(t) for t in toks}
                    h = [i for i in h if Q.normal_form(top[i]) in forms]
                entries.append(dict(t=top, r=int(ranks[ci]), nr=None, h=h, nh=[]))
            bl, br, bn = Q.best_of(layers, entries)
            items.append(dict(
                id=key + (f"#{concept}" if multi else ""), set="bench-fit", task=it["e"], difficulty=None,
                text=text, concept=concept, answer=it.get("target"), accepted=words, accepted_tokens=toks, near=[],
                note="", model=model, layers=entries, best_layer=bl, best_rank=br, best_near_rank=bn))
    meta = dict(lens=name, model=MODELS[tag], model_tag=tag, filter="f4", layers=layers, k_top=10, built=how,
                date=datetime.date.fromtimestamp(os.path.getmtime(src)).isoformat(),
                source=f"runs/fitA_{tag}_{mode}.json via scripts/qual/convert_fitA.py",
                sets={"bench-fit": len(items)})
    d = {"format": "qual-dump", "version": 1, "meta": meta, "items": items}
    out = f"{R}/runs/qual/{tag}/{Q.slug(name)}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = Q.save(out, d)
    print(f"wrote {out}: {len(items)} rows, {n} bytes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="9b,2b")
    a = ap.parse_args()
    for tag in a.tags.split(","):
        for mode in MODES:
            convert(tag, mode)


if __name__ == "__main__":
    main()
