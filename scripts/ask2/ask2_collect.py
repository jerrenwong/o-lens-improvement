"""ask2 step 1a: collect elicitation targets for three SPECIFIC-word few-shot formats (S0, S1, S2), same recipe as
scripts/e2_collect.py / m27_collect.py otherwise (same prefixes, same positions, batching by exact length, no padding).

9B: also recomputes X (prefix-token residual at the 7 readout layers) and the three OLD topic-format targets, since
    /root/e1/acts.pt no longer exists on the pod.
27B: X and the old targets are read later from .hf27/m27/acts_27b.pt; only the new targets are collected here.
Output: /workspace/o-lens-improvement/.hf27/ask2/acts_<tag>.pt {Y_new: [3,N,d] fp16, (9B) X, Y_old}
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = "27b" if "27B" in MODEL else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56]}[TAG]
OUT = f"{R}/.hf27/ask2/acts_{TAG}.pt"


def fmt(examples, q, a_lab):
    pre = "".join(f"{q}{t}\n{a_lab} {w}\n\n" for t, w in examples)
    return (pre + q, "\n" + a_lab)


NEW = [
    fmt([("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
         ("The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when", "Einstein"),
         ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
         ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")],
        "Passage: ", "Unstated word:"),
    fmt([("Its beam swept across the black water every ten seconds, warning the ships away from the rocks", "lighthouse"),
         ("The island nation whose capital is Reykjavik is known for its geysers and", "Iceland"),
         ("Die Geigen setzten ein, dann die Bläser, und der Dirigent hob den Stab, als", "orchestra"),
         ("He checked the knot, clipped the harness to the anchor and looked up at the wall of", "climbing")],
        "Text: ", "It never says the word, but it is about:"),
    fmt([("Her eyes itched, her nose ran, and she sneezed every time the cat jumped onto", "allergy"),
         ("The Russian novelist who wrote War and Peace also wrote a long novel about", "Tolstoy"),
         ("Elle a trié le linge par couleur, versé la lessive et lancé", "laundry"),
         ("The needle swung round and settled, pointing north, so the hikers turned left toward", "compass")],
        "Passage: ", "Hidden concept (one word):"),
]

sys.path.insert(0, f"{R}/scripts")
OLD = None
if TAG == "9b":
    import importlib.util
    sp = importlib.util.spec_from_file_location("e2c", f"{R}/scripts/e2_collect.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    OLD = m.ELICIT


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok = model.tokenizer
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model
    print("n_layers", len(model.layers), "d", d, "N", N, flush=True)

    def run_group(seqs, pos_layers, bs):
        res = {l: [] for l in pos_layers}
        for b0 in range(0, len(seqs), bs):
            ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
            rec = {}
            hk = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l))
                for l in pos_layers]
            try:
                model.forward(ids)
            finally:
                for h in hk:
                    h.remove()
            for l in pos_layers:
                res[l].append(rec[l].to(torch.float16))
        return {l: torch.cat(v) for l, v in res.items()}

    def batched(all_ids, pos_layers, bs):
        out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev) for l in pos_layers}
        groups = {}
        for i, s in enumerate(all_ids):
            groups.setdefault(len(s), []).append(i)
        for L, idx in sorted(groups.items()):
            r = run_group([all_ids[i] for i in idx], pos_layers, bs)
            ix = torch.tensor(idx, device=dev)
            for l in pos_layers:
                out[l][ix] = r[l]
        return out

    t0 = time.time()
    res = dict(new_formats=NEW)

    def targets(formats, name):
        Y = []
        for k, (pre, suf) in enumerate(formats):
            ids = [tok(pre + o["text"].strip("\n") + suf, add_special_tokens=False).input_ids for o in data]
            if k == 0:
                print(f"example {name}:", repr(tok.decode(ids[0])), flush=True)
            Y.append(batched(ids, [last], 64)[last].cpu())
            print(f"{name}{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / len(ids):.0f}", flush=True)
        return torch.stack(Y)

    if TAG == "9b":
        X = batched([o["ids"] for o in data], LAYERS, 256)
        res["X"] = {l: X[l].cpu() for l in LAYERS}
        del X
        print(f"X done {time.time() - t0:.0f}s", flush=True)
    res["Y_new"] = targets(NEW, "S")
    if TAG == "9b":
        res["Y_old"] = targets(OLD, "E")
    torch.save(res, OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
