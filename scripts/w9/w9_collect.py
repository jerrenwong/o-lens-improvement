"""w9 collect: passage activations X and answer states Y (topic + unstated-word questions, three wordings each) for a
row range of runs/w9/prefixes_100k.json. Same method as the existing collectors, whose prompt builders are imported
unchanged: scripts/ask3/ask3_collect9.py (topic; m2b_collect.py has identical wordings/builder) and
scripts/ask5/ask5_collect.py (unstated word; = s2b_collect.py), FAM "chat" (chat template, thinking off, few-shot,
assistant prefilled with the bare label). X = output of block l at the passage's last token (passage run alone),
as scripts/ask2/ask2_collect.py / e1_collect.py; Y = output of the LAST block at the prompt's last token. Batching by
exact length (no padding), forward stopped after the deepest needed block.
Usage: python w9_collect.py <model> --start S --end E --out DIR [--bs_x 256] [--bs_y 128]
       python w9_collect.py <model> --verify     (rows 2000..2199 of the OLD set vs the stored files; cosine)
Output (per part, resumable): DIR/X.pt {layer: [n,d] fp16}, DIR/Yt_w{k}.pt, DIR/Ys_w{k}.pt [n,d] fp16, DIR/timing.json
"""
import argparse
import copy
import importlib
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prefixes", default=f"{R}/runs/w9/prefixes_100k.json")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--bs_x", type=int, default=256)
ap.add_argument("--bs_y", type=int, default=128)
ap.add_argument("--verify", action="store_true")
ap.add_argument("--prefix_cache", action="store_true",
                help="Y prompts: run the token prefix shared by all prompts of a wording once and continue every batch "
                     "from a copy of its cache (same tokens; verified against the stored files)")
args = ap.parse_args()
M = args.model
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
sys.path.insert(0, f"{R}/src")
for d in ("ask3", "ask5"):
    sys.path.insert(0, f"{R}/scripts/{d}")
_argv = sys.argv
sys.argv = ["x", M, "chat"]
TOPIC = importlib.import_module("ask3_collect9")
SPEC = importlib.import_module("ask5_collect")
sys.argv = _argv
from workspace_lens import get_hf_model  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(8)
# the cuDNN SDPA backend re-plans for every new input shape (~2.5 s each; exact-length batching makes hundreds of
# shapes), so it is disabled; the flash / mem-efficient kernels are used instead (verified against the stored files)
torch.backends.cuda.enable_cudnn_sdp(False)
dev = "cuda"


class Stop(Exception):
    pass


def main():
    data = json.load(open(args.prefixes if not args.verify else f"{R}/runs/e1/prefixes.json"))
    if args.verify:
        args.start, args.end = 2000, 2200
        args.out = f"{R}/.hf27/w9/verify_{TAG}" + ("_pc" if args.prefix_cache else "")
    end = args.end or len(data)
    rows = data[args.start:end]
    os.makedirs(args.out, exist_ok=True)
    model = get_hf_model(M, attn_implementation="sdpa")
    tok = model.tokenizer
    last = len(model.layers) - 1
    n, d = len(rows), model.d_model
    print(f"[w9 collect] {M} rows {args.start}..{end} (n={n}) -> {args.out}", flush=True)

    def run_group(seqs, layers, bs, cache=None):
        res = {l: [] for l in layers}
        stop_at = max(layers)
        for b0 in range(0, len(seqs), bs):
            ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
            rec = {}

            def mk(l):
                def hook(m, i, oo):
                    rec[l] = (oo if torch.is_tensor(oo) else oo[0])[:, -1].to(torch.float16)
                    if l == stop_at:
                        raise Stop
                return hook
            hk = [model.layers[l].register_forward_hook(mk(l)) for l in layers]
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
                for h in hk:
                    h.remove()
            for l in layers:
                res[l].append(rec[l])
        return {l: torch.cat(v) for l, v in res.items()}

    def batched(all_ids, layers, bs, cache=None):
        out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev) for l in layers}
        groups = {}
        for i, s in enumerate(all_ids):
            groups.setdefault(len(s), []).append(i)
        done, tk = 0, time.time()
        for L, idx in sorted(groups.items()):
            r = run_group([all_ids[i] for i in idx], layers, bs, cache)
            done += len(idx)
            if done // 5000 != (done - len(idx)) // 5000:
                print(f"      {done}/{len(all_ids)} rows, {done / (time.time() - tk):.1f} rows/s", flush=True)
            ix = torch.tensor(idx, device=dev)
            for l in layers:
                out[l][ix] = r[l]
        return {l: v.cpu() for l, v in out.items()}

    timing = json.load(open(f"{args.out}/timing.json")) if os.path.exists(f"{args.out}/timing.json") else {}
    if not os.path.exists(f"{args.out}/X.pt"):
        ids = [o["ids"] for o in rows]
        t0 = time.time()
        X = batched(ids, LAYERS, args.bs_x)
        torch.cuda.synchronize(); dt = time.time() - t0
        ntok = sum(map(len, ids))
        timing["X"] = dict(seconds=dt, tokens=ntok, tok_per_s=ntok / dt, rows_per_s=n / dt)
        torch.save(X, f"{args.out}/X.pt")
        print(f"[w9 collect] X done {dt:.0f}s, {ntok / dt:.0f} tok/s", flush=True)
        json.dump(timing, open(f"{args.out}/timing.json", "w"), indent=1)
    for q, mod in (("t", TOPIC), ("s", SPEC)):
        for k in range(3):
            part = f"{args.out}/Y{q}_w{k}.pt"
            if os.path.exists(part):
                continue
            strs = [mod.build(tok, k, o["text"].strip("\n")) for o in rows]
            if k == 0 and args.start == 0:
                print(f"   example {q}{k}: {strs[0]!r}", flush=True)
            ids = [tok(s, add_special_tokens=False).input_ids for s in strs]
            t0 = time.time()
            if args.prefix_cache:
                P = 0
                while all(len(s) > P + 1 and s[P] == ids[0][P] for s in ids):
                    P += 1
                P -= 1  # keep at least the last shared token in the continuation
                out = model._text_module(input_ids=torch.tensor([ids[0][:P]], device=dev), use_cache=True)
                cache = out.past_key_values
                del out
                print(f"   prefix cache {q}{k}: {P} shared tokens of mean {sum(map(len, ids)) / n:.0f}", flush=True)
                y = batched([s[P:] for s in ids], [last], args.bs_y, cache)[last]
            else:
                y = batched(ids, [last], args.bs_y)[last]
            torch.cuda.synchronize(); dt = time.time() - t0
            ntok = sum(map(len, ids))
            timing[f"Y{q}{k}"] = dict(seconds=dt, tokens=ntok, tok_per_s=ntok / dt, rows_per_s=n / dt)
            torch.save(y, part)
            print(f"[w9 collect] Y{q}{k} done {dt:.0f}s, {ntok / dt:.0f} tok/s, {n / dt:.1f} rows/s", flush=True)
            json.dump(timing, open(f"{args.out}/timing.json", "w"), indent=1)
    if args.verify:
        verify(model)
    print("[w9 collect] done", flush=True)


def verify(model):
    """cosine of the new collection vs the stored arrays on old rows 2000..2199 (+ prompt-string check)."""
    s, e = 2000, 2200
    xf = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
    Xo = torch.load(xf, map_location="cpu")["X"]
    Xn = torch.load(f"{args.out}/X.pt")
    out = {}
    for l in LAYERS:
        c = torch.cosine_similarity(Xo[l][s:e].float(), Xn[l].float(), -1)
        out[f"X_L{l}"] = (c.min().item(), c.mean().item())
    del Xo
    for q, d_, mod in (("t", "ask3", TOPIC), ("s", "ask5", SPEC)):
        A = torch.load(f"{R}/.hf27/{d_}/acts_{TAG}.pt", map_location="cpu")
        data = json.load(open(f"{R}/runs/e1/prefixes.json"))
        for k in range(3):
            c = torch.cosine_similarity(A["Y_new"][k, s:e].float(), torch.load(f"{args.out}/Y{q}_w{k}.pt").float(), -1)
            out[f"Y{q}_w{k}"] = (c.min().item(), c.mean().item())
            out[f"prompt_{q}{k}_equal"] = mod.build(model.tokenizer, k, data[0]["text"].strip("\n")) == A["prompts"][k]
        del A
    for k_, v in out.items():
        print(f"[verify {TAG}] {k_}: {v}", flush=True)
    json.dump(out, open(f"{R}/runs/w9/verify_{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
