"""al3 step 2: generate sentences for the al3 concepts with the 9B writer (al2 recipe: T=0.9, top-p 0.95, top-k off,
stop at newline, prompts batched by exact token length, no padding). Styles S1-S5 (al2 prompts) + I2 (pilot prompt).
Target 8 accepted per (concept, style) (al3 filter: al2 cut + names + >= 6 tokens + pilot prefix rule + al2
near-duplicate rule); al2 sentences of al2 words count first for S1-S5. Passes: 0 = need + margin samples,
1..3 = top-up for jobs still short. Resumable (appends to gen_raw.jsonl; a pass is skipped for jobs already sampled in it).
Exits with code 3 if an ask3* tmux session appears (the chain waits and restarts).
Usage: gen.py [--dev cpu] [--tiny N] [--out FILE]"""
import argparse, json, math, os, random, sys, time, zlib
from collections import Counter, defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import (BIG, BIG2, MAXNEW, STYLES, TARGET, SNAME, Yield, check_yield, gen_prompt, load_model, log,
                        retry, run_adaptive, select)  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda")
ap.add_argument("--tiny", type=int, default=0)
ap.add_argument("--out", default=f"{BIG}/gen_raw.jsonl")
ap.add_argument("--seqs", type=int, default=144, help="start cap on sequences per generate call (halved on OOM)")
args = ap.parse_args()
torch.set_grad_enabled(False)
if args.dev == "cpu":
    torch.set_num_threads(128)
OUT = args.out
C = json.load(open(f"{BIG}/concepts.json"))
if args.tiny:
    C = [c for c in C if c["in_al2"]][: args.tiny // 2] + [c for c in C if not c["in_al2"]][: args.tiny - args.tiny // 2]
m2 = torch.load(f"{BIG2}/meta.pt", weights_only=False)
al2s = defaultdict(list)
for i, (w, k, s) in enumerate(zip(m2["words"], m2["style"], m2["sents"])):
    al2s[(w, k)].append((i, s))
del m2
jobs = [(c["word"], k) for c in C for k in STYLES]
model = load_model(args.dev)
hf, tok = model._hf_model, model.tokenizer
dev = args.dev
nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_nt = {}


def ntok(s):
    if s not in _nt:
        _nt[s] = len(tok(s, add_special_tokens=False).input_ids)
    return _nt[s]


def state():
    raw, passes = defaultdict(list), defaultdict(set)
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                o = json.loads(line)
            except Exception:
                continue
            raw[(o["word"], o["style"])].append(o["text"]); passes[(o["word"], o["style"])].add(o["pass"])
    acc, drops = {}, Counter()
    for (w, k) in jobs:
        items = [("al2", i, s, False) for i, s in al2s.get((w, k), [])] + [("new", n, t, True) for n, t in enumerate(raw[(w, k)])]
        acc[(w, k)] = len(select(w, k, items, ntok, drops=drops))
    return acc, passes, drops


def run(b, enc, nper, p):
    torch.manual_seed(zlib.crc32(f"al3|{p}|{b[0][0]}|{b[0][1]}|{len(b)}".encode()))
    ids = torch.tensor([enc[j] for j in b], device=dev)
    out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW[b[0][1]],
                      do_sample=True, temperature=0.9, top_p=0.95, top_k=0, num_return_sequences=nper,
                      pad_token_id=pad, eos_token_id=eos)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)


ST = dict(cap=args.seqs)
try:
    for p in range(4):
        acc, passes, drops = state()
        short = {j: TARGET - acc[j] for j in jobs if acc[j] < TARGET}
        log(f"pass {p}: jobs {len(jobs)}; short {len(short)}; missing sentences {sum(short.values())}; "
            f"accepted {sum(min(TARGET, a) for a in acc.values())}")
        todo = [j for j in short if p not in passes[j]]
        if not todo:
            continue
        nper_of = {j: (short[j] + max(2, math.ceil(0.25 * short[j]))) if p == 0 else min(12, 2 * short[j] + 1) for j in todo}
        enc = {j: tok(gen_prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in todo}
        groups = defaultdict(list)
        for j in todo:
            groups[(j[1], len(enc[j]), nper_of[j])].append(j)
        bs = []
        for (k, L, nper), js in sorted(groups.items()):
            per = max(1, args.seqs // nper)
            bs += [(js[i:i + per], nper) for i in range(0, len(js), per)]
        random.Random(p).shuffle(bs)
        t0, made = time.time(), 0
        log(f"pass {p}: todo jobs {len(todo)}, batches {len(bs)}, samples {sum(nper_of.values())}")
        with open(OUT, "a") as f:
            for bi, (b, nper) in enumerate(bs):
                if bi % 10 == 0:
                    check_yield(dev)
                texts = sum(run_adaptive(lambda ch: run(ch, enc, nper, p), b, ST, dev=dev, floor=12, unit=nper,
                                         name="generate"), [])
                for i, (w, k) in enumerate(b):
                    for t in texts[i * nper:(i + 1) * nper]:
                        f.write(json.dumps(dict(word=w, style=k, text=t, **{"pass": p})) + "\n")
                f.flush()
                made += len(texts)
                if bi % 25 == 0 or bi < 5 or bi == len(bs) - 1:
                    el = time.time() - t0
                    log(f"pass {p} batch {bi + 1}/{len(bs)} samples {made} {el:.0f}s ({made / max(el, 1e-9):.1f}/s); cap {ST['cap']} seqs; "
                        f"peak {torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0:.1f} GiB")
    acc, passes, drops = state()
    cnt = Counter(min(TARGET, a) for a in acc.values())
    log(f"final accepted {sum(min(TARGET, a) for a in acc.values())}; jobs by accepted count {dict(sorted(cnt.items()))}")
    log("drops " + json.dumps({f"{a}|{b}|{c}": v for (a, b, c), v in sorted(drops.items())}))
    print("GEN_DONE")
except Yield:
    log("ask3 session appeared: yielding the GPU (exit 3)")
    sys.exit(3)
