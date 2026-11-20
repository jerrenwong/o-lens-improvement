"""e1 step 1: label-free training prefixes (no benchmark text, no labels).

Sources: (a) WikiText-103 train, detokenised; (b) TinyStories; (c) Wikipedia in 10 languages (streaming).
Each source text -> one prefix of 5..48 tokens, starting at the text start (70%) or at a sentence start inside
it (30%), cut either at sentence-final punctuation ('sent') or at a random token ('rand'), 50/50.
Output: /root/e1/prefixes.json  [{src, kind, ids, text}]
"""
import json
import random
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/o-lens-improvement/scripts")
from detok_wikitext import detok_line  # noqa: E402

OUT = "/root/e1/prefixes.json"
N_WIKI, N_STORY, N_ML_PER = 13000, 12000, 550
LANGS = ["es", "fr", "de", "pt", "it", "ru", "tr", "pl", "nl", "ar"]
rng = random.Random(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
SENT_END = re.compile(r"[.!?。！？][\"'”»)]?\s*$")


def make_prefix(text):
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


out = []
# (a) WikiText-103 train, detokenised paragraphs (no headings)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
rows = [r for r in ds["text"] if r.strip() and not r.strip().startswith("=") and len(r) > 200]
rng.shuffle(rows)
got = 0
for r in rows:
    p = make_prefix(detok_line(r.strip()))
    if p:
        out.append(dict(src="wiki_en", **p))
        got += 1
    if got >= N_WIKI:
        break
print("wiki_en", got, flush=True)
# (b) TinyStories
try:
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    got = 0
    for i, ex in enumerate(ds):
        if i % 7:  # spread over the stream
            continue
        p = make_prefix(ex["text"])
        if p:
            out.append(dict(src="story", **p))
            got += 1
        if got >= N_STORY:
            break
    print("story", got, flush=True)
except Exception as e:  # noqa: BLE001
    print("TinyStories FAILED", repr(e), flush=True)
# (c) Wikipedia, non-English
for lang in LANGS:
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        got = 0
        for i, ex in enumerate(ds):
            if i % 5:
                continue
            paras = [q for q in ex["text"].split("\n") if len(q) > 120]
            if not paras:
                continue
            p = make_prefix(rng.choice(paras[:3]))
            if p:
                out.append(dict(src=f"wiki_{lang}", **p))
                got += 1
            if got >= N_ML_PER:
                break
        print(lang, got, flush=True)
    except Exception as e:  # noqa: BLE001
        print(lang, "FAILED", repr(e)[:200], flush=True)

# drop anything overlapping benchmark prompt text (fit+dev prompt texts only; no labels used)
try:
    rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json"))
    RV = rd["vocab"]
    bt = ["".join(RV[x] for x in it["toks"]) for it in rd["items"] if it["s"] in ("fit", "dev")]
    frags = {f.strip() for s in bt for f in re.split(r"[.!?\n]", s) if len(f.strip()) >= 25}
    n0 = len(out)
    out = [o for o in out if not any(f in o["text"] for f in frags)]
    print(f"benchmark-overlap filter ({len(frags)} fragments of fit+dev prompts): dropped {n0 - len(out)}")
except Exception as e:  # noqa: BLE001
    print("overlap check skipped", repr(e))

rng.shuffle(out)
json.dump(out, open(OUT, "w"))
from collections import Counter  # noqa: E402
print("total", len(out), Counter(o["src"] for o in out), Counter(o["kind"] for o in out))
L = [len(o["ids"]) for o in out]
print("len mean", sum(L) / len(L), "min", min(L), "max", max(L))
for o in out[:20]:
    print(f"  [{o['src']:7s} {o['kind']}] {o['text']!r}")
