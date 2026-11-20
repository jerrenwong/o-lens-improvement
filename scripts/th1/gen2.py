"""th1 step 1: 60 new sentences per word (30 per g1 style), T=0.9, top-p 0.95, <=40 tokens. Resumable."""
import json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from th1_common import R, BIG, log, retry, load_model
sys.path.insert(0, f"{R}/scripts")
from g1_gen import prompt  # noqa: E402

OUT = f"{BIG}/gen_raw.jsonl"
NPER, PER_BATCH = 30, 4
torch.set_grad_enabled(False)
words = [o["word"] for o in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "words.json")))]
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            o = json.loads(line); done.add((o["word"], o["style"]))
        except Exception:
            pass
todo = [(w, k) for w in words for k in (0, 1) if (w, k) not in done]
log(f"jobs todo {len(todo)} done {len(done)}")
if not todo:
    print("ALL_DONE"); sys.exit()
model = load_model()
hf, tok = model._hf_model, model.tokenizer
nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc = {j: tok(prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in todo}
groups = {}
for j in todo:
    groups.setdefault(len(enc[j]), []).append(j)
batches = [js[i:i + PER_BATCH] for _, js in sorted(groups.items()) for i in range(0, len(js), PER_BATCH)]


def run(b):
    torch.manual_seed(zlib.crc32(f"th1|{b[0][0]}|{b[0][1]}".encode()))
    ids = torch.tensor([enc[j] for j in b], device="cuda")
    out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=True,
                      temperature=0.9, top_p=0.95, top_k=0, num_return_sequences=NPER, pad_token_id=pad,
                      eos_token_id=eos)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)


t0 = time.time()
with open(OUT, "a") as f:
    for bi, b in enumerate(batches):
        texts = retry(run, b)
        for i, (w, k) in enumerate(b):
            for t in texts[i * NPER:(i + 1) * NPER]:
                f.write(json.dumps(dict(word=w, style=k, text=t)) + "\n")
        f.flush()
        if bi % 10 == 0:
            log(f"batch {bi + 1}/{len(batches)} {time.time() - t0:.0f}s")
print("ALL_DONE")
