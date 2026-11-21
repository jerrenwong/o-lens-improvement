"""sub2 step 1: label every al2 sentence (.hf27/al2/meta.pt order) with the sub1 pilot stage-3 measurement:
rank = best rank of the hidden word's single-token surface forms forms(w) in the 9B model's next-token logits after the
few-shot "Unstated word:" question (Q_HEAD), on the sentence (rank) and on its word-shuffled version (rank_shuf).
Rank over the FULL vocabulary (the pilot excludes no tokens in stage 3): rank = 1 + #tokens with logit > best form.
Functions Q_HEAD / shuffled / forms / last_logits / by_len copied verbatim from scripts/sub1/pilot_v1.py (importing it
would run the whole pilot). Batches of exactly equal token length, size 96, no padding (as the pilot).
Resumable: shards in .hf27/sub2/label_shards/. Output .hf27/sub2/al2_ranks.pt + pilot agreement check.
Usage: label.py [--n N] [--out DIR]   (N > 0: test on N sentences incl. pilot-overlap rows)."""
import argparse, glob, json, os, random, re, sys, time, zlib
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
_argv, sys.argv = sys.argv, sys.argv[:1]  # al2_common imports g1_gen, which parses sys.argv
from al2_common import log, free_mb  # noqa: E402
sys.argv = _argv

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--out", default=f"{R}/.hf27/sub2")
ap.add_argument("--bs", type=int, default=96)
args = ap.parse_args()
OUT = args.out
SH = f"{OUT}/label_shards"
os.makedirs(SH, exist_ok=True)
torch.set_grad_enabled(False)

# ---- copied from pilot_v1.py (stage 3) ----
Q_SHOTS = [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
           ("The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when", "Einstein"),
           ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
           ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]
Q_HEAD = "".join(f"Passage: {t}\nUnstated word: {a}\n\n" for t, a in Q_SHOTS)
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def shuffled(sent):
    ws = WORD_RE.findall(sent)
    random.Random(zlib.crc32(sent.encode())).shuffle(ws)
    return " ".join(ws)


def by_len(items, enc, bs):
    g = {}
    for it in items:
        g.setdefault(len(enc[it]), []).append(it)
    return [v[i:i + bs] for _, v in sorted(g.items()) for i in range(0, len(v), bs)]
# --------------------------------------------


meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
words, sents, styles = meta["words"], meta["sents"], meta["style"]
N = len(words)
pilot = [json.loads(l) for l in open(f"{R}/.hf27/sub1/pilot.jsonl")]
key = {(w, k, s): i for i, (w, k, s) in enumerate(zip(words, styles, sents))}
pil = {}
for r in pilot:
    if r["src"] == "new":
        continue
    i = key.get((r["word"], int(r["style"][1:]) - 1, r["sent"]))
    if i is not None:
        pil[i] = r
rows = list(range(N))
if args.n:
    ov = sorted(pil)[: args.n // 2]
    rest = random.Random(0).sample([i for i in rows if i not in pil], args.n - len(ov))
    rows = sorted(ov + rest)
log(f"sentences {N}; measuring {len(rows)}; pilot rows in meta {len(pil)}")

done = {}
for f in sorted(glob.glob(f"{SH}/*.pt")):
    try:
        d = torch.load(f)
    except Exception as e:
        log(f"bad shard {f}: {e}")
        continue
    done.update(d)
todo_keys = [(i, k) for i in rows for k in (0, 1) if (i, k) not in done]
log(f"done {len(done)} prompts from shards; todo {len(todo_keys)}")

if todo_keys:
    NEED = 36000
    while True:
        fm = free_mb()
        if fm >= NEED:
            break
        log(f"waiting for GPU: free {fm} MiB < {NEED}")
        time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(0.42)
    from workspace_lens import get_hf_model
    while True:
        try:
            model = get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log("OOM at model load; waiting 60 s")
            time.sleep(60)
    hf, tok = model._hf_model, model.tokenizer
    log(f"model loaded; free {free_mb()} MiB")

    def last_logits(ids):
        x = torch.tensor(ids, device="cuda")
        return hf(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=1).logits[:, -1].float()

    def forms(w):
        out = set()
        for v in {w, w.lower(), w[:1].upper() + w[1:]}:
            for f in (" " + v, v):
                ids = tok(f, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out.add(ids[0])
        return sorted(out)

    FO = {w: forms(w) for w in set(words[i] for i, _ in todo_keys)}
    nof = [w for w, f in FO.items() if not f]
    log(f"words without single-token form: {len(nof)} {nof[:10]}")
    t0 = time.time()
    qenc = {}
    for i, k in todo_keys:
        s = sents[i] if k == 0 else shuffled(sents[i])
        qenc[(i, k)] = tok(Q_HEAD + f"Passage: {s}\nUnstated word:", add_special_tokens=False).input_ids
    batches = by_len(todo_keys, qenc, args.bs)
    log(f"tokenized {len(qenc)} prompts in {time.time() - t0:.0f}s; batches {len(batches)}; "
        f"max len {max(len(v) for v in qenc.values())}")

    def run(b):
        while True:
            try:
                return last_logits([qenc[x] for x in b])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"CUDA OOM (batch {len(b)} x {len(qenc[b[0]])}); free {free_mb()} MiB; retry in 60 s")
                time.sleep(60)

    buf, t0 = {}, time.time()
    for bi, b in enumerate(batches):
        lg = run(b)
        top = lg.topk(3, -1).indices.tolist()
        for row, x, tp in zip(lg, b, top):
            fi = FO[words[x[0]]]
            if not fi:
                buf[x] = (lg.shape[1] + 1, tp)
                continue
            vals = row[fi]
            rank = int((row[None, :] > vals[:, None]).sum(-1).min().item()) + 1
            buf[x] = (rank, tp)
        del lg
        if len(buf) >= 20000 or bi == len(batches) - 1:
            torch.save(buf, f"{SH}/s{int(time.time() * 1000)}_{bi}.pt")
            done.update(buf)
            buf = {}
        if bi % 100 == 0:
            el = time.time() - t0
            log(f"batch {bi + 1}/{len(batches)} len {len(qenc[b[0]])} {el:.0f}s eta {el / (bi + 1) * (len(batches) - bi - 1):.0f}s")
    del model, hf
    torch.cuda.empty_cache()

rank = torch.full((N,), -1, dtype=torch.int64)
rank_shuf = torch.full((N,), -1, dtype=torch.int64)
top3 = torch.full((N, 3), -1, dtype=torch.int64)
top3_shuf = torch.full((N, 3), -1, dtype=torch.int64)
for (i, k), (r, tp) in done.items():
    (rank if k == 0 else rank_shuf)[i] = r
    (top3 if k == 0 else top3_shuf)[i] = torch.tensor(tp)
have = (rank > 0) & (rank_shuf > 0)
log(f"labelled {int(have.sum())} of {N}")
# pilot agreement on overlapping rows
cmp = [i for i in pil if have[i]]
eq = sum(int(rank[i]) == pil[i]["rank"] for i in cmp)
eqs = sum(int(rank_shuf[i]) == pil[i]["rank_shuf"] for i in cmp)


def b(r, rs):
    return "NOT" if r > 10 else "GIVE" if rs <= 10 else "SUB"


eqb = sum(b(int(rank[i]), int(rank_shuf[i])) == b(pil[i]["rank"], pil[i]["rank_shuf"]) for i in cmp)
dif = [(i, int(rank[i]), pil[i]["rank"], int(rank_shuf[i]), pil[i]["rank_shuf"]) for i in cmp
       if int(rank[i]) != pil[i]["rank"] or int(rank_shuf[i]) != pil[i]["rank_shuf"]]
chk = dict(n_compared=len(cmp), rank_equal=eq, rank_shuf_equal=eqs, bin_equal=eqb, n_diff=len(dif), diffs=dif[:30])
log(f"pilot check {json.dumps({k: v for k, v in chk.items() if k != 'diffs'})}; first diffs {dif[:10]}")
if args.n == 0:
    assert bool(have.all()), "missing labels"
torch.save(dict(rank=rank, rank_shuf=rank_shuf, top3=top3, top3_shuf=top3_shuf, labelled=have,
                note="rank over full vocab, best of forms(w); bins: NOT rank>10, GIVE rank<=10 & rank_shuf<=10, SUB else"),
           f"{OUT}/al2_ranks.pt")
os.makedirs(f"{R}/scripts/sub2/results", exist_ok=True)
json.dump(chk, open(f"{R}/scripts/sub2/results/pilot_check{'_test' if args.n else ''}.json", "w"), indent=1)
print("LABEL_DONE")
