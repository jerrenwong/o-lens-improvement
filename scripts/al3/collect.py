"""al3 step 3b: hidden states of every NEW al3 sentence, run alone (tokenizer, no special tokens, as al2 collect.py),
forward hooks on model.layers[l] at LAYERS: (i) last token -> X_L{l}.pt, (ii) mean over all tokens -> Xmean_L{l}.pt
(as mp1_collect.py), fp16, rows in meta new_idx order. Batches of equal token length (128). Resumable by chunks
of 8,192 sentences (.hf27/al3/collect_chunks/). Reused al2 rows: .hf27/al2/X_L*.pt and .hf27/mp1/Xmean_L*.pt.
Usage: collect.py [--dev cpu] [--meta FILE] [--out DIR] [--n N]"""
import argparse, glob, os, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import BIG, LAYERS, Yield, check_yield, load_model, log, run_adaptive  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda")
ap.add_argument("--meta", default=f"{BIG}/meta.pt")
ap.add_argument("--out", default=BIG)
ap.add_argument("--n", type=int, default=0)
args = ap.parse_args()
dev = args.dev
if dev == "cpu":
    torch.set_num_threads(128)
torch.set_grad_enabled(False)
CH = f"{args.out}/collect_chunks"
os.makedirs(CH, exist_ok=True)
CS = 8192
sents = torch.load(args.meta, weights_only=False)["new_sents"]
if args.n:
    sents = sents[: args.n]
N = len(sents)
chunks = [(c0, min(N, c0 + CS)) for c0 in range(0, N, CS)]
todo = [c for c in chunks if not os.path.exists(f"{CH}/c{c[0]}_{c[1]}.pt")]
log(f"new sentences {N}; chunks {len(chunks)}; todo {len(todo)}")
ST = dict(cap=128)
try:
    if todo:
        model = load_model(dev)
        tok = model.tokenizer
        d = model.d_model

        def one(ids_list):
            ids = torch.tensor(ids_list, device=dev)
            rec = {}
            hk = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])))(l)) for l in LAYERS]
            try:
                model.forward(ids)
            finally:
                for h in hk:
                    h.remove()
            return {l: (rec[l][:, -1].to(torch.float16).cpu(), rec[l].float().mean(1).to(torch.float16).cpu())
                    for l in LAYERS}

        t0 = time.time()
        for c0, c1 in todo:
            check_yield(dev)
            ids = [tok(s, add_special_tokens=False).input_ids for s in sents[c0:c1]]
            Xl = {l: torch.zeros(c1 - c0, d, dtype=torch.float16) for l in LAYERS}
            Xm = {l: torch.zeros(c1 - c0, d, dtype=torch.float16) for l in LAYERS}
            groups = defaultdict(list)
            for i, s in enumerate(ids):
                groups[len(s)].append(i)
            for L, idx in sorted(groups.items()):
                for b0 in range(0, len(idx), 128):
                    sub = idx[b0:b0 + 128]
                    parts_ = run_adaptive(lambda ch: (ch, one([ids[i] for i in ch])), sub, ST, dev=dev, name="collect")
                    for ch, r in parts_:
                        ti = torch.tensor(ch)
                        for l in LAYERS:
                            Xl[l][ti], Xm[l][ti] = r[l]
            torch.save(dict(last=Xl, mean=Xm), f"{CH}/c{c0}_{c1}.pt")
            log(f"chunk {c0}-{c1} done {time.time() - t0:.0f}s; cap {ST['cap']}")
except Yield:
    log("ask3 session appeared: yielding the GPU (exit 3)")
    sys.exit(3)

parts = [torch.load(f"{CH}/c{c0}_{c1}.pt") for c0, c1 in chunks]
for l in LAYERS:
    torch.save(torch.cat([p["last"][l] for p in parts]), f"{args.out}/X_L{l}.pt")
    torch.save(torch.cat([p["mean"][l] for p in parts]), f"{args.out}/Xmean_L{l}.pt")
log(f"saved X_L*/Xmean_L* ({N} rows)")
print("COLLECT_DONE")
