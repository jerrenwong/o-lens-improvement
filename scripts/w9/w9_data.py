"""w9 part 1: extend the e1 passage set (runs/e1/prefixes.json, 30,500) to 100,000 with the SAME public sources and
the SAME construction rules (make_prefix is a verbatim copy of scripts/e1_data.py), new passages to train only.

Sources (public Hugging Face datasets only; no other text):
  wiki_en : Salesforce/wikitext, wikitext-103-raw-v1, split train. Same row filter + rng=Random(0) shuffle as e1. The
            e1 loop is REPLAYED (same rng, same tokenizer) to find the shuffled rows e1 consumed and to check that the
            replay reproduces e1's wiki_en passages; new passages continue from the next shuffled row.
  story   : roneneldan/TinyStories, split train, streaming; e1 used rows i % 7 == 0, w9 uses i % 7 == 1 (disjoint rows).
  wiki_xx : wikimedia/wikipedia, 20231101.<lang>, split train, streaming; e1 used articles i % 5 == 0, w9 i % 5 == 1.
Counts: e1 proportions scaled to 100,000 total -> new wiki_en 29,626, story 27,344, 1,253 per language (12,530).
Exclusions (prompt text only, never labels): every prompt of every benchmark set on disk (v1 incl. order-ops, v2, v3,
v4, genword; plus the 9B readout-viewer token strings) ->
  (A) e1 rule: drop if a prompt fragment (split at .!?\\n, >= 25 chars) occurs in the passage;
  (B) reverse: drop if a passage fragment (same split, >= 25 chars) occurs in a prompt;
  (C) drop if the passage shares any 8-word shingle (lower-cased \\w+ words) with a prompt.
  Also exact-text duplicates of any old passage or earlier new passage are dropped.
Output: runs/w9/prefixes_100k.json (first 30,500 = runs/e1/prefixes.json verbatim, then the new passages shuffled),
        runs/w9/data_prov.json (per new passage: source row index), runs/w9/data_stats.json.
"""
import glob
import json
import random
import re
import sys
import time
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts")
from detok_wikitext import detok_line  # noqa: E402

SMOKE = "--smoke" in sys.argv
OUT = f"{R}/runs/w9/prefixes_100k.json" if not SMOKE else f"{R}/runs/w9/smoke_prefixes.json"
N_WIKI_OLD = 13000
NEW = dict(wiki_en=29626, story=27344, ml=1253)
if SMOKE:
    NEW = dict(wiki_en=200, story=200, ml=20)
SURPLUS = 1.12  # generate a little more, trim to exact counts after the exclusion filters
LANGS = ["es", "fr", "de", "pt", "it", "ru", "tr", "pl", "nl", "ar"]
rng = random.Random(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
SENT_END = re.compile(r"[.!?。！？][\"'”»)]?\s*$")


def make_prefix(text):  # verbatim from scripts/e1_data.py
    text = text.strip()
    if rng.random() < 0.3:  # start at a sentence start inside the text
        starts = [m.end() for m in re.finditer(r"[.!?]\s+(?=[A-ZÀ-ÿА-Я\"'])", text)]
        if starts:
            text = text[rng.choice(starts):]
    ids = tok(text, add_special_tokens=False).input_ids[:80]
    if len(ids) < 6:
        return None
    strs = tok.convert_ids_to_tokens(ids)
    dec = [tok.decode([i]) for i in ids]
    kind = "sent" if rng.random() < 0.5 else "rand"
    if kind == "sent":
        ends = [p + 1 for p in range(4, min(48, len(ids))) if SENT_END.search(dec[p])
                and (p + 1 == len(ids) or dec[p + 1][:1] in (" ", "\n", ""))]
        if not ends:
            kind = "rand"
        else:
            n = rng.choice(ends)
    if kind == "rand":
        hi = min(48, len(ids) - 1) if len(ids) > 5 else len(ids)
        if hi < 5:
            return None
        n = rng.randint(5, hi)
    del strs
    return dict(kind=kind, ids=ids[:n], text=tok.decode(ids[:n]))


old = json.load(open(f"{R}/runs/e1/prefixes.json"))
old_text = {o["text"] for o in old}
old_wiki = {o["text"] for o in old if o["src"] == "wiki_en"}
stats = dict(sources={}, old_counts=Counter(o["src"] for o in old))
t0 = time.time()
new, prov = [], []

# ------------------------------------------------------------------ (a) WikiText-103: replay e1, then continue
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
rows_all = ds["text"]
rows = [(j, r) for j, r in enumerate(rows_all) if r.strip() and not r.strip().startswith("=") and len(r) > 200]
order = list(range(len(rows)))
# e1 shuffled the list of row strings; shuffling the index list with the same rng state gives the same permutation
rng.shuffle(order)
got, k, replay = 0, 0, []
while got < N_WIKI_OLD:
    p = make_prefix(detok_line(rows[order[k]][1].strip()))
    if p:
        replay.append(p["text"]); got += 1
    k += 1
match = len(set(replay) & old_wiki)
print(f"[w9 data] wikitext replay: e1 consumed {k} shuffled rows; replayed {len(replay)} passages, {match} equal to e1's "
      f"{len(old_wiki)} wiki_en passages; {time.time() - t0:.0f}s", flush=True)
stats["wikitext_replay"] = dict(rows_filtered=len(rows), e1_consumed_shuffled_rows=k, replay_match=match,
                                old_wiki=len(old_wiki))
k0, got = k, 0
while got < NEW["wiki_en"] * SURPLUS and k < len(order):
    j, r = rows[order[k]]
    p = make_prefix(detok_line(r.strip()))
    if p:
        new.append(dict(src="wiki_en", **p)); prov.append(dict(src="wiki_en", row=j, shuf_pos=k)); got += 1
    k += 1
stats["sources"]["wiki_en"] = dict(dataset="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train",
                                   shuffled_positions=[k0, k], generated=got)
print(f"[w9 data] wiki_en new {got} (shuffled positions {k0}..{k - 1}); {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ (b) TinyStories, rows i % 7 == 1
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
got, last = 0, None
for i, ex in enumerate(ds):
    if i % 7 != 1:
        continue
    p = make_prefix(ex["text"])
    if p:
        new.append(dict(src="story", **p)); prov.append(dict(src="story", row=i)); got += 1
    last = i
    if got >= NEW["story"] * SURPLUS:
        break
stats["sources"]["story"] = dict(dataset="roneneldan/TinyStories", config=None, split="train (streaming)",
                                 rows="i % 7 == 1", row_range=[1, last], generated=got)
print(f"[w9 data] story new {got} (rows 1..{last}, i%7==1); {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ (c) Wikipedia, articles i % 5 == 1
for lang in LANGS:
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    got, last = 0, None
    for i, ex in enumerate(ds):
        if i % 5 != 1:
            continue
        paras = [q for q in ex["text"].split("\n") if len(q) > 120]
        last = i
        if not paras:
            continue
        p = make_prefix(rng.choice(paras[:3]))
        if p:
            new.append(dict(src=f"wiki_{lang}", **p)); prov.append(dict(src=f"wiki_{lang}", row=i, id=ex.get("id"))); got += 1
        if got >= NEW["ml"] * SURPLUS:
            break
    stats["sources"][f"wiki_{lang}"] = dict(dataset="wikimedia/wikipedia", config=f"20231101.{lang}",
                                            split="train (streaming)", rows="i % 5 == 1", row_range=[1, last],
                                            generated=got)
    print(f"[w9 data] wiki_{lang} new {got} (articles 1..{last}, i%5==1); {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ benchmark prompts (prompt text only)
prompts = []
files = (glob.glob("/root/jpp_lens/data/jlens/evaluations/lens-eval-*.json") + glob.glob(f"{R}/data/evals_v*/lens-eval-*.json")
         + glob.glob(f"{R}/data/evals_gen/lens-eval-*.json"))
for f in sorted(files):
    d = json.load(open(f))
    it = d["items"] if isinstance(d, dict) else d
    for x in it:
        pr = x["prompt"]
        prompts.append(pr if isinstance(pr, str) else json.dumps(pr, ensure_ascii=False))
for f in glob.glob("/workspace/fleet/projects/o-lens/site/data/readouts_*.json"):
    rd = json.load(open(f))
    RV = rd["vocab"]
    prompts += ["".join(RV[x] for x in it["toks"]) for it in rd["items"]]
stats["benchmark"] = dict(files=sorted(files), n_prompts=len(prompts))
print(f"[w9 data] {len(prompts)} benchmark prompts from {len(files)} files + readout viewers", flush=True)
frags = {f.strip() for s in prompts for f in re.split(r"[.!?\n]", s) if len(f.strip()) >= 25}
big = "\x00".join(prompts)
W = re.compile(r"\w+")


def shingles(s, n=8):
    w = W.findall(s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


SH = set().union(*[shingles(s) for s in prompts])


KEY = 25
fr_by_key = {}
for f in frags:
    fr_by_key.setdefault(f[:KEY], []).append(f)
big_keys = {big[i:i + KEY] for i in range(len(big) - KEY + 1)}


def why(text):
    if any(f in text for i in range(len(text) - KEY + 1) for f in fr_by_key.get(text[i:i + KEY], ())):
        return "A"
    if any(len(f.strip()) >= KEY and f.strip()[:KEY] in big_keys and f.strip() in big for f in re.split(r"[.!?\n]", text)):
        return "B"
    if shingles(text) & SH:
        return "C"
    return None


drops = Counter()
seen = set(old_text)
keep_new, keep_prov = [], []
for o, pv in zip(new, prov):
    if o["text"] in seen:
        drops["dup"] += 1
        continue
    w = why(o["text"])
    if w:
        drops[w] += 1
        continue
    seen.add(o["text"])
    keep_new.append(o); keep_prov.append(pv)
old_flag = Counter(why(o["text"]) for o in old)
print(f"[w9 data] dropped from new: {dict(drops)}; (info) old passages flagged by the extended rules: "
      f"{ {k: v for k, v in old_flag.items() if k} }", flush=True)
stats["dropped_new"] = dict(drops)
stats["old_flagged_info"] = {str(k): v for k, v in old_flag.items() if k}
# trim to exact counts per source (generation order)
cap = {"wiki_en": NEW["wiki_en"], "story": NEW["story"], **{f"wiki_{l}": NEW["ml"] for l in LANGS}}
cnt = Counter()
fin, finp = [], []
for o, pv in zip(keep_new, keep_prov):
    if cnt[o["src"]] < cap[o["src"]]:
        cnt[o["src"]] += 1; fin.append(o); finp.append(pv)
short = {s: cap[s] - cnt[s] for s in cap if cnt[s] < cap[s]}
print(f"[w9 data] final new counts {dict(cnt)}; short {short}", flush=True)
for s in stats["sources"]:
    rr = [p["row"] for p in finp if p["src"] == s]
    stats["sources"][s]["used"] = len(rr)
    stats["sources"][s]["used_row_min_max"] = [min(rr), max(rr)] if rr else None
perm = list(range(len(fin)))
random.Random(9).shuffle(perm)
fin = [fin[i] for i in perm]; finp = [finp[i] for i in perm]
allp = old + fin
assert allp[:len(old)] == old
json.dump(allp, open(OUT, "w"))
json.dump(finp, open(OUT.replace("prefixes_100k.json", "data_prov.json").replace("smoke_prefixes", "smoke_prov"), "w"))
L = [len(o["ids"]) for o in fin]
stats.update(total=len(allp), new=len(fin), new_counts=dict(cnt), kind=dict(Counter(o["kind"] for o in fin)),
             len_mean=sum(L) / len(L), len_min=min(L), len_max=max(L), seconds=time.time() - t0)
json.dump(stats, open(OUT.replace("prefixes_100k.json", "data_stats.json").replace("smoke_prefixes", "smoke_stats"), "w"),
          indent=1, default=str)
print(json.dumps(stats, indent=1, default=str)[:3000])
for o in fin[:10]:
    print(f"  [{o['src']:7s} {o['kind']}] {o['text']!r}")
