"""Self-test of the probe-set code paths of compare.py / table.py (no GPU): writes a 6-probe sample file and two FAKE
qual dumps of it (hand-made top lists, not lens outputs), then checks the counts compare.py reports.

    python3 scripts/qual/selftest.py

Writes runs/qual/probes_sample.json and runs/qual/test/{fake_a,fake_b}.json (model tag "test"; never publish them).
Covers: whole-token hit, piece-only hit ([x~]), near mark, a control (no accepted), an unscorable probe
(scorable false), is_next true / false / unknown, improved / worsened / both-miss.
The page's JavaScript is tested on the same files by scripts/qual/test_page.js.
Exit code 0 = all checks passed.
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import qualfmt as Q  # noqa: E402
import table  # noqa: E402

R = "/workspace/o-lens-improvement"
PROBES = [
    dict(id="obj-01", category="implied-object", difficulty="medium",
         text="I am thirsty, the sun is scorching. Should I bite through this thorny thing?",
         accepted=["cactus"], near=["desert", "plant"], note="multi-token word"),
    dict(id="obj-02", category="implied-object", difficulty="easy",
         text="The door is locked and I have patted every pocket twice. Where on earth did I put my",
         accepted=["keys", "key"], near=["lock"]),
    dict(id="emo-01", category="implied-emotion", difficulty="hard",
         text="She read the letter twice, folded it very small and stared out of the window for an hour.",
         accepted=["sadness", "grief"], near=["letter"]),
    dict(id="bridge-01", category="bridge-entity", difficulty="medium",
         text="The ocean on the coast of the country where Carnival is most famously celebrated is the",
         accepted=["Brazil"], near=["Rio"]),
    dict(id="garb-01", category="garbled-word", difficulty="hard", text="Zhyxqv wrotkle.",
         accepted=["жых"], near=[], note="no usable token form"),
    dict(id="control-01", category="control", difficulty="easy",
         text="This section describes the changes made since the previous version.", accepted=[], near=[]),
]
LAYERS = [4, 8, 12]


def ent(top, r, nr=None, h=(), nh=(), hp=()):
    e = dict(t=top, r=r, nr=nr, h=list(h), nh=list(nh))
    if hp:
        e["hp"] = list(hp)
    return e


def fake(lens, spec):
    items = []
    for p in PROBES:
        layers = spec[p["id"]]
        it = dict(id=p["id"], set="probes", task=p["category"], difficulty=p["difficulty"] if p["accepted"] else None,
                  text=p["text"], concept=(p["accepted"] or [None])[0], answer=None, accepted=p["accepted"],
                  accepted_tokens=[], near=p["near"], note=p.get("note", ""), model={"next": ["x"], "knows": ["y"]},
                  forms=[], scorable=p["id"] not in ("garb-01", "control-01"),
                  is_next={"obj-02": True, "bridge-01": False, "obj-01": False, "emo-01": None}.get(p["id"]),
                  layers=layers)
        it["best_layer"], it["best_rank"], it["best_near_rank"] = Q.best_of(LAYERS, layers)
        items.append(it)
    meta = dict(lens=lens, model="test/fake", model_tag="test", filter="f4", layers=LAYERS, k_top=20, built="fake",
                date="2026-11-27", source="selftest.py", sets={"probes": len(items)})
    return {"format": "qual-dump", "version": 1, "meta": meta, "items": items}


W = ["the", "a", "of", "and", "in", "to", "it", "is", "was", "on"]


def top(*front):
    return list(front) + W[: 10 - len(front)]


def main():
    os.makedirs(f"{R}/runs/qual/test", exist_ok=True)
    Q.save(f"{R}/runs/qual/probes_sample.json", {"version": 1, "categories": {}, "probes": PROBES})
    miss = lambda r: ent(top(" sun"), r)  # noqa: E731
    A = {"obj-01": [miss(300), miss(200), miss(150)],
         "obj-02": [miss(40), ent(top(" pocket", " keys"), 2, h=[1]), miss(30)],
         "emo-01": [miss(900), miss(800), miss(700)],
         "bridge-01": [miss(50), miss(60), miss(70)],
         "garb-01": [ent(top(), None), ent(top(), None), ent(top(), None)],
         "control-01": [ent(top(" version"), None), ent(top(), None), ent(top(), None)]}
    B = {"obj-01": [miss(90), ent(top(" desert", " cact"), 2, nr=1, h=[1], nh=[0], hp=[1]), miss(20)],
         "obj-02": [miss(40), miss(25), miss(30)],
         "emo-01": [miss(500), miss(400), miss(300)],
         "bridge-01": [ent(top(" Brazil"), 1, h=[0]), miss(60), miss(70)],
         "garb-01": [ent(top(), None), ent(top(), None), ent(top(), None)],
         "control-01": [ent(top(" changes"), None), ent(top(), None), ent(top(), None)]}
    pa, pb = f"{R}/runs/qual/test/fake_a.json", f"{R}/runs/qual/test/fake_b.json"
    Q.save(pa, fake("Fake A", A))
    Q.save(pb, fake("Fake B", B))
    da, db = Q.load(pa), Q.load(pb)
    rows = compare.pair(da, db, "probes", "best", 10)
    c = compare.counts(rows)
    want = dict(n=4, hitA=1, hitB=2, improved=2, worsened=1, control=1, unscorable=1)
    bad = {k: (c[k], v) for k, v in want.items() if c[k] != v}
    assert not bad, f"counts {bad}"
    nx = compare.by(rows, "is_next")
    assert nx["next word"]["worsened"] == 1 and nx["not next word"]["improved"] == 2 and nx["unknown"]["n"] == 1, nx
    r1 = next(r for r in rows if r["a"]["id"] == "obj-01")
    marks = Q.mark_top(r1["eb"])
    assert marks[:2] == ["(desert)", "[cact~]"], marks
    buf = io.StringIO()
    with redirect_stdout(buf):
        compare.main([pa, pb, "--md"])
        compare.main([pa, pb, "--next", "no"])
        table.main([pa, pb, pa])
    out = buf.getvalue()
    for s in ("| next word | 1 | 1 | 0 | 0 | 1 |", "| not next word | 2 | 0 | 2 | 2 | 0 |", "[cact~]",
              "2 unscorable" if False else "1 unscorable", "| is_next: not next word | 2 | 0 | 2 | 0 |"):
        assert s in out, f"missing {s!r} in output"
    print(out)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
