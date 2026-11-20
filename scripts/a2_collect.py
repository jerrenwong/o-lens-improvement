"""a2 (PREREG_v3, recipe A'): recollect ONLY the elicitation targets Y_k whose format triggered the whitespace rule
(.hf27/a2/rule.json from scripts/a2_ytop.py before). Identical to scripts/m27_collect.py except: X is not recollected
(reused from .hf27/m27/acts_27b.pt), and for a triggered format the rule's whitespace token id is appended to the
tokenized elicitation prompt; Y_k = output of the LAST block at the new last position. Untriggered formats keep the
old Y_k. Output: .hf27/a2/Y_ws.pt {Y: [3,N,d] fp16, changed: [k...], rule}
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
OUT = "/workspace/o-lens-improvement/.hf27/a2"
ELICIT = [
    # E0: "Passage / One-word topic", own examples (encyclopedic, cut mid-sentence, French, story)
    ("Passage: The tide pulled the little boat away from the dock while the fisherman untangled his nets.\n"
     "One-word topic: fishing\n\n"
     "Passage: The senate voted late on Tuesday to approve the new budget, despite\n"
     "One-word topic: politics\n\n"
     "Passage: Elle a sorti la farine, les œufs et le beurre pour préparer\n"
     "One-word topic: baking\n\n"
     "Passage: Lily saw the old dog shivering in the rain, so she opened the door and\n"
     "One-word topic: kindness\n\n"
     "Passage: ", "\nOne-word topic:"),
    # E1: "Text / Key concept (one word)", different examples
    ("Text: Ash and molten rock poured down the slopes as the villagers fled toward the coast.\n"
     "Key concept (one word): volcano\n\n"
     "Text: The company reported a sharp rise in quarterly profits after it cut\n"
     "Key concept (one word): business\n\n"
     "Text: Der Arzt hörte das Herz des Kindes ab und\n"
     "Key concept (one word): medicine\n\n"
     "Text: Tom put on his coat and boots and stepped out into the deep white\n"
     "Key concept (one word): snow\n\n"
     "Text: ", "\nKey concept (one word):"),
    # E2: instruction header, arrow format, different examples
    ("Name the main subject of each text in one word.\n\n"
     "The orchestra tuned their instruments as the conductor raised his baton.\n"
     "=> Subject: music\n\n"
     "In 1492 the ships left the port of Palos and sailed west into\n"
     "=> Subject: exploration\n\n"
     "O jogador recebeu o cartão vermelho e saiu de campo\n"
     "=> Subject: football\n\n"
     "She typed the last line of code, pressed enter, and watched the program\n"
     "=> Subject: programming\n\n", "\n=> Subject:"),
]



def main():
    torch.set_grad_enabled(False)
    rule = json.load(open(f"{OUT}/rule.json"))
    changed = [r["k"] for r in rule if r["triggered"]]
    print("rule", rule, "changed", changed, flush=True)
    data = json.load(open("/workspace/o-lens-improvement/runs/e1/prefixes.json"))
    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok, hf = model.tokenizer, model._hf_model
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model
    print("n_layers", len(model.layers), "d", d, "N", N, flush=True)

    def run_group(seqs, pos_layers, bs):
        res = {l: [] for l in pos_layers}
        for b0 in range(0, len(seqs), bs):
            chunk = seqs[b0:b0 + bs]
            ids = torch.tensor([s for _, s in chunk], device=dev)
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
            r = run_group([(i, all_ids[i]) for i in idx], pos_layers, bs)
            ix = torch.tensor(idx, device=dev)
            for l in pos_layers:
                out[l][ix] = r[l]
        return out

    old = torch.load("/workspace/o-lens-improvement/.hf27/m27/acts_27b.pt")["Y"]
    t0 = time.time()
    Y = []
    for k, (pre, suf) in enumerate(ELICIT):
        if k not in changed:
            Y.append(old[k])
            print(f"Y{k} kept (rule not triggered)", flush=True)
            continue
        app = rule[k]["append_id"]
        ids = [tok(pre + o["text"].strip("\n") + suf, add_special_tokens=False).input_ids + [app] for o in data]
        # check: the first 1000 old-style prompts reproduce the stored Y (consistency of this code with m27_collect)
        if k == changed[0]:
            chk = [s[:-1] for s in ids[1000:1064]]
            yo = batched(chk, [last], 64)[last].float().cpu()
            cs = torch.cosine_similarity(yo, old[k][1000:1064].float(), -1)
            print(f"check: recomputed old Y{k} rows 1000:1064 cos min {cs.min():.4f} mean {cs.mean():.4f}", flush=True)
        y = batched(ids, [last], 64)[last]
        Y.append(y.cpu())
        print(f"Y{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / len(ids):.0f}; appended id {app}", flush=True)
        print("example elicitation tail:", repr(tok.decode(ids[1000][-40:])), flush=True)
    torch.save(dict(Y=torch.stack(Y).cpu(), changed=changed, rule=rule), f"{OUT}/Y_ws.pt")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
