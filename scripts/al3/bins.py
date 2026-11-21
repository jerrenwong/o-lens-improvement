"""al3: label-bin shares (inferable = rank <= 10; given away = rank <= 10 and rank_shuf <= 10; subtle = inferable and
not given away; NOT = rank > 10) by type x style for the NEW al3 sentences, for the reused al2 sentences inside al3,
and for al2 as a whole (all words / nameable words), by style. Writes scripts/al3/results/bins.json + prints tables."""
import json, sys
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import R, BIG, BIG2, RES, SNAME  # noqa: E402

TYPES = ["PERSON", "PLACE", "THING", "EVENT", "STATE"]
m3 = torch.load(f"{BIG}/meta.pt", weights_only=False)
L3 = torch.load(f"{BIG}/labels_new.pt")
L2 = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt")
m2 = torch.load(f"{BIG2}/meta.pt", weights_only=False)
judged = {r["word"]: r for r in json.load(open(f"{R}/.hf27/sub1/al2_words_judged.json"))}


def b(rk, rs):
    return "NOT" if rk > 10 else "GIVE" if rs <= 10 else "SUB"


acc = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # acc[src][(type, style)][bin]
for r in m3["rows"]:
    if r["src"] == "new":
        rk, rs = int(L3["rank"][r["new_idx"]]), int(L3["rank_shuf"][r["new_idx"]])
    else:
        rk, rs = int(L2["rank"][r["al2_idx"]]), int(L2["rank_shuf"][r["al2_idx"]])
    x = b(rk, rs)
    src = "al3 new" if r["src"] == "new" else "al3 reused al2"
    for t in (r["type"], "ALL"):
        for s in (SNAME[r["style"]], "ALL"):
            acc[src][(t, s)][x] += 1
            acc["al3 all"][(t, s)][x] += 1
for i, (w, k) in enumerate(zip(m2["words"], m2["style"])):
    x = b(int(L2["rank"][i]), int(L2["rank_shuf"][i]))
    for s in (SNAME[k], "ALL"):
        acc["al2 all words"][("ALL", s)][x] += 1
        if judged.get(w, {}).get("keep"):
            acc["al2 nameable (judged keep)"][("ALL", s)][x] += 1
out = {}
for src, tab in acc.items():
    out[src] = {}
    for (t, s), c in sorted(tab.items()):
        n = sum(c.values())
        out[src][f"{t}|{s}"] = dict(n=n, inferable=round((c["GIVE"] + c["SUB"]) / n, 3), given_away=round(c["GIVE"] / n, 3),
                                    subtle=round(c["SUB"] / n, 3))
json.dump(out, open(f"{RES}/bins.json", "w"), indent=1)
for src, tab in out.items():
    print(f"== {src}: inferable / given away / subtle (n)")
    ts = ["ALL"] + [t for t in TYPES if any(k.startswith(t + "|") for k in tab)]
    print("        " + "".join(f"{s:>20s}" for s in ["ALL", "S1", "S2", "S3", "S4", "S5", "I2"]))
    for t in ts:
        cells = []
        for s in ["ALL", "S1", "S2", "S3", "S4", "S5", "I2"]:
            v = tab.get(f"{t}|{s}")
            cells.append(f"{v['inferable']:.2f}/{v['given_away']:.2f}/{v['subtle']:.2f} {v['n']:>5d}" if v else "-")
        print(f"{t:8s}" + "".join(f"{c:>20s}" for c in cells))
print("BINS_DONE")
