"""ask6 collect: ask5_collect.py method (chat template, thinking off, stop after the last block, batch 64, resumable per
wording) with the format variant chosen in step 1 (scripts/ask6/ask6_formats.py: build(), argv[3] = i/ii/iii/iv).
For iii: when the raw top-1 at the label position is a format token, that token is appended and the state at the new
last position is used. Output .hf27/ask6/acts_<tag>.pt {Y_new [3,N,d] fp16, prompts, variant}.
Usage: python ask6_collect.py <model> chat <variant>"""
import importlib.util
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
MODEL, FAM, VAR = sys.argv[1], sys.argv[2], sys.argv[3]
TAG = "27b" if "27B" in MODEL else "9b"
OUT = f"{R}/.hf27/ask6/acts_{TAG}.pt"
sv = sys.argv
sys.argv = ["x", MODEL]
sp = importlib.util.spec_from_file_location("f6", f"{R}/scripts/ask6/ask6_formats.py")
f6 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(f6)
sys.argv = sv
W = f6.W


def build(tok, k, text):
    return f6.build(tok, k, text, "i" if VAR == "iii" else VAR)


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok = model.tokenizer
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model
    V = model.unembed(torch.zeros(1, d, device=dev)).shape[-1]
    strs = [tok.decode([i]) for i in range(V)]
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
    fmt_tok = torch.tensor([s.strip() != "" and not re.search(r"[^\W_]", s) and i not in special
                            for i, s in enumerate(strs)], device=dev)

    class Stop(Exception):
        pass

    def run_group(seqs, bs):
        res = []
        for b0 in range(0, len(seqs), bs):
            ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
            rec = {}

            def hook(m, i, oo):
                rec["y"] = (oo if torch.is_tensor(oo) else oo[0])[:, -1]
                raise Stop

            hk = model.layers[last].register_forward_hook(hook)
            try:
                model.forward(ids)
            except Stop:
                pass
            finally:
                hk.remove()
            res.append(rec["y"].to(torch.float16))
        return torch.cat(res)

    def batched(ids, bs):
        out = torch.zeros(len(ids), d, dtype=torch.float16, device=dev)
        groups = {}
        for i, s in enumerate(ids):
            groups.setdefault(len(s), []).append(i)
        done, tk = 0, time.time()
        for L, idx in sorted(groups.items()):
            out[torch.tensor(idx, device=dev)] = run_group([ids[i] for i in idx], bs)
            done += len(idx)
            if done // 3000 != (done - len(idx)) // 3000:
                print(f"   {done}/{len(ids)} {done / (time.time() - tk):.1f} prefixes/s", flush=True)
        return out

    t0 = time.time()
    Y, prompts = [], []
    BS = int(os.environ.get("BS", "64"))
    for k in range(3):
        part = f"{R}/.hf27/ask6/Y_{TAG}_w{k}.pt"
        strs_p = [build(tok, k, o["text"].strip("\n")) for o in data]
        prompts.append(strs_p[0])
        print(f"W{k} example ({VAR}): {strs_p[0]!r}", flush=True)
        if os.path.exists(part):
            Y.append(torch.load(part))
            print(f"W{k} loaded from {part}", flush=True)
            continue
        ids = [tok(s, add_special_tokens=False).input_ids for s in strs_p]
        out = batched(ids, BS)
        if VAR == "iii":
            top = torch.cat([model.unembed(out[i:i + 256].float()).argmax(-1) for i in range(0, N, 256)])
            fi = torch.nonzero(fmt_tok[top])[:, 0].tolist()
            print(f"W{k} appended format token on {len(fi)}/{N}", flush=True)
            if fi:
                o2 = batched([ids[i] + [int(top[i])] for i in fi], BS)
                out[torch.tensor(fi, device=dev)] = o2
        Y.append(out.cpu())
        torch.save(Y[-1], part)
        print(f"W{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / N:.0f}", flush=True)
    torch.save(dict(Y_new=torch.stack(Y), prompts=prompts, fam=FAM, variant=VAR, wordings=W), OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
