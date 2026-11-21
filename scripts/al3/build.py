"""al3 step 2b: build the al3 sentence set from al2 sentences (al2 words, S1-S5, in al2 order) + gen_raw.jsonl with the
shared select() rule (<= 8 per concept x style). Writes .hf27/al3/meta.pt:
rows: word, style (0..4 = S1..S5, 5 = I2), sent, src ('al2'|'new'), al2_idx (row of .hf27/al2/meta.pt or -1),
new_idx (row of the al3 new-sentence arrays or -1), split (concept split: train/val/test/near), type;
new_sents (sentences needing label + collect, in new_idx order); concepts; drop counts.
Usage: build.py [--gen FILE] [--out FILE] [--tiny N]"""
import argparse, json, sys
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import BIG, BIG2, RES, STYLES, SNAME, TARGET, log, select  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=f"{BIG}/gen_raw.jsonl")
ap.add_argument("--out", default=f"{BIG}/meta.pt")
ap.add_argument("--tiny", type=int, default=0)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
C = json.load(open(f"{BIG}/concepts.json"))
if args.tiny:
    C = [c for c in C if c["in_al2"]][: args.tiny // 2] + [c for c in C if not c["in_al2"]][: args.tiny - args.tiny // 2]
cinfo = {c["word"]: c for c in C}
m2 = torch.load(f"{BIG2}/meta.pt", weights_only=False)
al2s = defaultdict(list)
for i, (w, k, s) in enumerate(zip(m2["words"], m2["style"], m2["sents"])):
    al2s[(w, k)].append((i, s))
raw = defaultdict(list)
n_raw = Counter()
for line in open(args.gen):
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o["word"] in cinfo:
        raw[(o["word"], o["style"])].append(o["text"]); n_raw[(SNAME[o["style"]], o["pass"])] += 1
_nt = {}


def ntok(s):
    if s not in _nt:
        _nt[s] = len(tok(s, add_special_tokens=False).input_ids)
    return _nt[s]


drops = Counter()
rows, new_sents = [], []
short = Counter()
for c in C:
    w = c["word"]
    for k in STYLES:
        items = [("al2", i, s, False) for i, s in al2s.get((w, k), [])] + [("new", n, t, True) for n, t in enumerate(raw[(w, k)])]
        kept = select(w, k, items, ntok, drops=drops)
        if len(kept) < TARGET:
            short[(SNAME[k], len(kept))] += 1
        for src, pay, s in kept:
            r = dict(word=w, style=k, sent=s, src=src, al2_idx=pay if src == "al2" else -1, new_idx=-1,
                     split=c["split"], type=c["type"])
            if src == "new":
                r["new_idx"] = len(new_sents); new_sents.append(s)
            rows.append(r)
al2_avail = sum(len(al2s.get((c["word"], k), [])) for c in C for k in range(5))
log(f"concepts {len(C)}; rows {len(rows)} (al2 {sum(r['src'] == 'al2' for r in rows)} of {al2_avail} available, "
    f"new {len(new_sents)}); raw new samples {sum(n_raw.values())}")
log(f"short (style, kept) {dict(sorted(short.items()))}")
log(f"drops {dict(sorted(drops.items()))}")
tab = defaultdict(Counter)
for r in rows:
    tab[r["split"]][SNAME[r["style"]]] += 1
log("rows by split x style " + json.dumps({k: dict(v) for k, v in tab.items()}))
meta = dict(rows=rows, new_sents=new_sents, concepts=C, drops={"|".join(k): v for k, v in drops.items()},
            n_raw={f"{a}|p{b}": v for (a, b), v in n_raw.items()}, short={f"{a}|{b}": v for (a, b), v in short.items()})
torch.save(meta, args.out)
if not args.tiny:
    json.dump(dict(n_rows=len(rows), n_new=len(new_sents), al2_used=sum(r['src'] == 'al2' for r in rows),
                   al2_available=al2_avail, raw=meta["n_raw"], drops=meta["drops"], short=meta["short"],
                   by_split_style={k: dict(v) for k, v in tab.items()}), open(f"{RES}/build.json", "w"), indent=1)
print("BUILD_DONE")
