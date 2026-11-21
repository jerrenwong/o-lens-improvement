"""al3 step 3a: label every NEW al3 sentence with the sub2/label.py measurement (code copied): rank of the hidden
word's best single-token form (forms rule) in the 9B next-token logits after the few-shot "Unstated word:" prompt
(Q_HEAD), full vocabulary, on the sentence (rank) and on its word-shuffled version (rank_shuf); top-3 token ids.
Batches of exactly equal token length (no padding), size 96. Resumable shards in .hf27/al3/label_shards/.
Reused al2 sentences keep their .hf27/sub2/al2_ranks.pt labels (fit.py joins them).
Usage: label.py [--dev cpu] [--meta FILE] [--out DIR] [--n N]"""
import argparse, glob, os, sys, time
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import BIG, Q_HEAD, Yield, check_yield, forms_tok, load_model, log, run_adaptive, shuffled  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda")
ap.add_argument("--meta", default=f"{BIG}/meta.pt")
ap.add_argument("--out", default=BIG)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=96)
args = ap.parse_args()
dev = args.dev
if dev == "cpu":
    torch.set_num_threads(128)
SH = f"{args.out}/label_shards"
os.makedirs(SH, exist_ok=True)
torch.set_grad_enabled(False)
meta = torch.load(args.meta, weights_only=False)
sents = meta["new_sents"]
N = len(sents)
wordof = [None] * N
for r in meta["rows"]:
    if r["new_idx"] >= 0:
        wordof[r["new_idx"]] = r["word"]
rows = list(range(N if not args.n else min(N, args.n)))
done = {}
for f in sorted(glob.glob(f"{SH}/*.pt")):
    try:
        done.update(torch.load(f))
    except Exception as e:
        log(f"bad shard {f}: {e}")
todo = [(i, k) for i in rows for k in (0, 1) if (i, k) not in done]
log(f"new sentences {N}; measuring {len(rows)}; done {len(done)} prompts; todo {len(todo)}")


def by_len(items, enc, bs):
    g = {}
    for it in items:
        g.setdefault(len(enc[it]), []).append(it)
    return [v[i:i + bs] for _, v in sorted(g.items()) for i in range(0, len(v), bs)]


buf = {}
ST = dict(cap=args.bs)
try:
    if todo:
        model = load_model(dev)
        hf, tok = model._hf_model, model.tokenizer

        def last_logits(ids):
            x = torch.tensor(ids, device=dev)
            return hf(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=1).logits[:, -1].float()

        FO = {w: forms_tok(tok, w) for w in set(wordof[i] for i, _ in todo)}
        assert all(FO.values()), [w for w, f in FO.items() if not f][:10]
        qenc = {}
        for i, k in todo:
            s = sents[i] if k == 0 else shuffled(sents[i])
            qenc[(i, k)] = tok(Q_HEAD + f"Passage: {s}\nUnstated word:", add_special_tokens=False).input_ids
        batches = by_len(todo, qenc, args.bs)
        log(f"batches {len(batches)}; max len {max(len(v) for v in qenc.values())}")
        buf, t0 = {}, time.time()
        for bi, b in enumerate(batches):
            if bi % 20 == 0:
                check_yield(dev)
            lg = torch.cat(run_adaptive(lambda ch: last_logits([qenc[x] for x in ch]), b, ST, dev=dev, name="label"))
            top = lg.topk(3, -1).indices.tolist()
            for row, x, tp in zip(lg, b, top):
                vals = row[FO[wordof[x[0]]]]
                buf[x] = (int((row[None, :] > vals[:, None]).sum(-1).min().item()) + 1, tp)
            del lg
            if len(buf) >= 20000 or bi == len(batches) - 1:
                torch.save(buf, f"{SH}/s{int(time.time() * 1000)}_{bi}.pt")
                done.update(buf); buf = {}
            if bi % 100 == 0:
                el = time.time() - t0
                log(f"batch {bi + 1}/{len(batches)} cap {ST['cap']} {el:.0f}s eta {el / (bi + 1) * (len(batches) - bi - 1):.0f}s")
except Yield:
    if buf:
        torch.save(buf, f"{SH}/s{int(time.time() * 1000)}_y.pt")
    log("ask3 session appeared: yielding the GPU (exit 3)")
    sys.exit(3)

rank = torch.full((N,), -1, dtype=torch.int64)
rank_shuf = torch.full((N,), -1, dtype=torch.int64)
top3 = torch.full((N, 3), -1, dtype=torch.int64)
top3_shuf = torch.full((N, 3), -1, dtype=torch.int64)
for (i, k), (r, tp) in done.items():
    if i < N:
        (rank if k == 0 else rank_shuf)[i] = r
        (top3 if k == 0 else top3_shuf)[i] = torch.tensor(tp)
have = (rank > 0) & (rank_shuf > 0)
log(f"labelled {int(have.sum())} of {N}")
if not args.n:
    assert bool(have.all()), "missing labels"
torch.save(dict(rank=rank, rank_shuf=rank_shuf, top3=top3, top3_shuf=top3_shuf, labelled=have,
                note="new al3 sentences (meta new_idx order); rank over full vocab, best of forms(w); bins: NOT rank>10, "
                     "GIVE rank<=10 & rank_shuf<=10, SUB else"), f"{args.out}/labels_new.pt")
print("LABEL_DONE")
