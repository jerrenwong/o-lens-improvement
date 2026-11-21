"""al2 step 1: generate sentences, 5 styles, 3 per (word, style), T=0.9, top-p 0.95 (al1 recipe). Resumable.
S1/S2 for al1's 2,400 words are reused from al1's gen_raw (not generated again).
pass 0: 3 samples per job. After >= 2,000 sentences: projected time; if > 90 min, remaining jobs get 2 samples and
the flag file CUT2 is written (collect then keeps 2 per style for every word).
pass 1: top-up of 3 more samples for jobs with < 3 usable outputs (cut ok, word not named, >= 6 tokens)."""
import json, os, random, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al2_common import BIG, MAXNEW, log, retry, load_model, prompt, cut, names

os.makedirs(BIG, exist_ok=True)
OUT = f"{BIG}/gen_raw.jsonl"
CUT2 = f"{BIG}/CUT2"
PER_BATCH = 48
torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))
words = json.load(open(f"{HERE}/words.json"))
al1w = set(json.load(open(f"{HERE}/al1_words.json")))
jobs = [(w, k) for w in words for k in range(5) if not (k < 2 and w in al1w)]
model = load_model()
hf, tok = model._hf_model, model.tokenizer
nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def usable(w, k, t):
    s = cut(t, k)
    return s is not None and not names(s, w) and len(tok(s, add_special_tokens=False).input_ids) >= 6


def state():
    got, n = {}, 0
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                o = json.loads(line)
            except Exception:
                continue
            key = (o["word"], o["style"])
            g = got.setdefault(key, [set(), 0])
            g[0].add(o["pass"]); g[1] += usable(o["word"], o["style"], o["text"]); n += 1
    return got, n


def batches_for(todo):
    enc = {j: tok(prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in todo}
    groups = {}
    for j in todo:
        groups.setdefault((j[1], len(enc[j])), []).append(j)
    bs = [js[i:i + PER_BATCH] for _, js in sorted(groups.items()) for i in range(0, len(js), PER_BATCH)]
    random.Random(0).shuffle(bs)  # mixed styles/lengths, so early timing is representative
    return bs, enc


def run(b, enc, nper, p):
    torch.manual_seed(zlib.crc32(f"al2|{p}|{b[0][0]}|{b[0][1]}".encode()))
    ids = torch.tensor([enc[j] for j in b], device="cuda")
    out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW[b[0][1]],
                      do_sample=True, temperature=0.9, top_p=0.95, top_k=0, num_return_sequences=nper,
                      pad_token_id=pad, eos_token_id=eos)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)


for p in (0, 1):
    got, n0 = state()
    if p == 0:
        todo = [j for j in jobs if 0 not in got.get(j, [set()])[0]]
    else:
        todo = [j for j in jobs if got.get(j, [set(), 0])[1] < 3 and 1 not in got[j][0]]
    nper = 2 if os.path.exists(CUT2) else 3
    log(f"pass {p}: jobs {len(jobs)} todo {len(todo)}; lines already {n0}; samples per job {nper}")
    if not todo:
        continue
    bs, enc = batches_for(todo)
    t0, made, reported = time.time(), 0, False
    with open(OUT, "a") as f:
        for bi, b in enumerate(bs):
            texts = retry(run, b, enc, nper, p)
            for i, (w, k) in enumerate(b):
                for t in texts[i * nper:(i + 1) * nper]:
                    f.write(json.dumps(dict(word=w, style=k, text=t, p=p, **{"pass": p})) + "\n")
            f.flush()
            made += len(texts)
            if p == 0 and not reported and made >= 2000:
                el = time.time() - t0
                proj = el / made * len(todo) * nper
                log(f"TIMING: first {made} sentences in {el:.0f}s ({made / el:.1f}/s); projected pass-0 total "
                    f"{proj / 60:.1f} min for {len(todo) * nper} sentences")
                reported = True
                if proj > 90 * 60:
                    open(CUT2, "w").write(f"projected {proj / 60:.1f} min\n")
                    nper = 2
                    log("projected > 90 min: CUT to 2 samples per job for the remaining jobs")
            if bi % 25 == 0:
                log(f"pass {p} batch {bi + 1}/{len(bs)} sentences {made} {time.time() - t0:.0f}s")
    log(f"pass {p} generation time {time.time() - t0:.0f}s, sentences {made}")
print("ALL_DONE")
