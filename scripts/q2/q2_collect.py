"""q2 collect: answer states Y (output of the LAST block at the prompt's last token) for the new A-lens questions
q3..q7 (scripts/q2/q2_qs.py, 3 wordings each) over rows of runs/w9/prefixes_100k.json. Same method as
scripts/w9/w9_collect.py (batching by exact length, forward stopped after the last block, optional shared-prefix KV
cache: the chat/few-shot prefix shared by all prompts of a wording runs once and every batch continues from a copy).
Passage activations X are NOT recollected (stored: .hf27/ask2/acts_9b.pt | .hf27/pod_root/e1/acts_2b.pt for rows
0..30499, .hf27/w9/<tag>_new/X.pt for rows 30500..99999).
Usage: python q2_collect.py <model> [--qs q3,q4,q5,q6,q7] [--wordings 0,1,2] [--start 0 --end 100000] [--prefix_cache]
Order: wording-major (all questions' w0 first). Output .hf27/q2/<tag>/Y_<q>_w<k>.pt [n, d] fp16 (resumable per file),
runs/q2/collect_<tag>.json (rates)."""
import argparse
import copy
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/q2")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--qs", default="q3,q4,q5,q6,q7")
ap.add_argument("--wordings", default="0,1,2")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=100000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--prefix_cache", action="store_true")
ap.add_argument("--verify", type=int, default=256, help="rows checked: prefix cache vs plain forward (q3 w0)")
args = ap.parse_args()
M = args.model
TAG = "2b" if "2B" in M else "9b"
import q2_qs  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.cuda.enable_cudnn_sdp(False)  # as w9_collect (cuDNN SDPA re-plans per shape)
dev = "cuda"
OUT = f"{R}/.hf27/q2/{TAG}"
os.makedirs(OUT, exist_ok=True)


class Stop(Exception):
    pass


def main():
    q2_qs.hygiene()
    SP = q2_qs.specs()
    rows = json.load(open(f"{R}/runs/w9/prefixes_100k.json"))[args.start:args.end]
    n = len(rows)
    model = get_hf_model(M, attn_implementation="sdpa")
    tok = model.tokenizer
    last = len(model.layers) - 1
    d = model.d_model

    def run_group(seqs, bs, cache=None):
        res = []
        for b0 in range(0, len(seqs), bs):
            ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
            rec = {}

            def hook(m, i, oo):
                rec["y"] = (oo if torch.is_tensor(oo) else oo[0])[:, -1].to(torch.float16)
                raise Stop
            hk = model.layers[last].register_forward_hook(hook)
            try:
                if cache is None:
                    model.forward(ids)
                else:
                    c = copy.deepcopy(cache)
                    c.reorder_cache(torch.zeros(ids.shape[0], dtype=torch.long, device=dev))
                    model._text_module(input_ids=ids, past_key_values=c, use_cache=True)
            except Stop:
                pass
            finally:
                hk.remove()
            res.append(rec["y"])
        return torch.cat(res)

    def collect(ids, use_cache):
        out = torch.zeros(len(ids), d, dtype=torch.float16, device=dev)
        cache, P = None, 0
        if use_cache:
            while all(len(s) > P + 1 and s[P] == ids[0][P] for s in ids):
                P += 1
            P -= 1
            o = model._text_module(input_ids=torch.tensor([ids[0][:P]], device=dev), use_cache=True)
            cache = o.past_key_values
            del o
        groups = {}
        for i, s in enumerate(ids):
            groups.setdefault(len(s), []).append(i)
        done, tk = 0, time.time()
        for L, idx in sorted(groups.items()):
            out[torch.tensor(idx, device=dev)] = run_group([ids[i][P:] for i in idx], args.bs, cache)
            done += len(idx)
            if done // 10000 != (done - len(idx)) // 10000:
                print(f"      {done}/{len(ids)} rows, {done / (time.time() - tk):.1f} rows/s", flush=True)
        return out.cpu(), P

    timing = {}
    tf = f"{R}/runs/q2/collect_{TAG}.json"
    if os.path.exists(tf):
        timing = json.load(open(tf))
    if args.prefix_cache and args.verify:
        strs = [q2_qs.prompt(tok, SP["q3"][0], o["text"].strip("\n")) for o in rows[:args.verify]]
        ids = [tok(s, add_special_tokens=False).input_ids for s in strs]
        a, P = collect(ids, True)
        b, _ = collect(ids, False)
        c = torch.cosine_similarity(a.float(), b.float(), -1)
        timing["verify_prefix_cache"] = dict(rows=args.verify, shared_tokens=P, cos_min=c.min().item(),
                                             cos_mean=c.mean().item())
        print(f"[q2 collect] verify prefix cache vs plain ({args.verify} rows, {P} shared tokens): cos min "
              f"{c.min().item():.5f} mean {c.mean().item():.5f}", flush=True)
        assert c.min().item() > 0.99
    for k in map(int, args.wordings.split(",")):
        for q in args.qs.split(","):
            part = f"{OUT}/Y_{q}_w{k}.pt"
            if os.path.exists(part):
                continue
            strs = [q2_qs.prompt(tok, SP[q][k], o["text"].strip("\n")) for o in rows]
            print(f"   example {q} w{k}: {strs[0]!r}", flush=True)
            ids = [tok(s, add_special_tokens=False).input_ids for s in strs]
            t0 = time.time()
            y, P = collect(ids, args.prefix_cache)
            dt = time.time() - t0
            ntok = sum(map(len, ids))
            timing[f"{q}_w{k}"] = dict(rows=n, seconds=dt, tokens=ntok, shared_prefix=P, rows_per_s=n / dt)
            torch.save(y, part)
            print(f"[q2 collect] {q} w{k}: {n} rows {dt:.0f}s, {n / dt:.1f} rows/s (prefix {P} of mean "
                  f"{ntok / n:.0f} tokens)", flush=True)
            json.dump(timing, open(tf, "w"), indent=1)
    print("[q2 collect] done", flush=True)


if __name__ == "__main__":
    main()
