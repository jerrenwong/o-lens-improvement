"""m27 copy of e2_collect.py for Qwen/Qwen3.6-27B (changes marked "m27"; recipe unchanged).
e2 (= e1 step 2, model as argv[1]; same recipe): activations at the prefix's last token (prefix run alone) and three few-shot elicitation targets.

X[l]  = output of block l (== benchmark resid[l] == hidden_states[l+1]) at the prefix's last token, l in 4..28.
Y[k]  = output of the LAST block (pre final norm; model.unembed applies norm + head) at the last token of
        elicitation prompt k (k = 0,1,2), which wraps the full prefix in a one-word few-shot format.
No padding: sequences are batched by exact length (hybrid-attention model).
Output: /root/e1/acts.pt {X: {l: [N,d] fp16}, Y: [3,N,d] fp16, meta}
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = "27b" if "27B" in MODEL else ("2b" if "2B" in MODEL else "9b")  # m27
LAYERS = {"2b": [3, 6, 9, 12, 15, 18, 21], "9b": [4, 8, 12, 16, 20, 24, 28],
          "27b": [8, 16, 24, 32, 40, 48, 56]}[TAG]  # m27: 27B readout layers
OUT27 = "/workspace/o-lens-improvement/.hf27/m27"  # m27: /root has ~6 GB free; outputs go to the volume
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
    data = json.load(open("/workspace/o-lens-improvement/runs/e1/prefixes.json"))  # m27: repo copy (same md5 as /root/e1)
    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok, hf = model.tokenizer, model._hf_model
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model
    print("n_layers", len(model.layers), "d", d, "N", N, flush=True)

    # ---- convention checks on 3 prefixes
    for o in data[:3]:
        ids = torch.tensor([o["ids"]], device=dev)
        out = hf(ids, output_hidden_states=True)
        hs = out.hidden_states
        rec = {}
        hk = [model.layers[l].register_forward_hook(
            (lambda l: lambda m, i, oo: rec.__setitem__(l, oo if torch.is_tensor(oo) else oo[0]))(l))
            for l in (LAYERS[2], LAYERS[5], last)]
        model.forward(ids)
        for h in hk:
            h.remove()
        c = [torch.cosine_similarity(rec[l][0, -1].float(), hs[l + 1][0, -1].float(), 0).item() for l in (LAYERS[2], LAYERS[5])]
        lg = model.unembed(rec[last][:, -1]).float()
        ref = out.logits[0, -1].float()
        print(f"check: cos(block l out, hidden_states[l+1]) L{LAYERS[2]} {c[0]:.4f} L{LAYERS[5]} {c[1]:.4f}; "
              f"unembed(last block) vs logits max|diff| {(lg[0, :ref.shape[0]] - ref).abs().max().item():.3f} "
              f"argmax equal {lg[0, :ref.shape[0]].argmax().item() == ref.argmax().item()}", flush=True)

    def run_group(seqs, pos_layers, bs):
        """seqs: list of (row, ids) all same length -> {layer: [n, d]} at last token."""
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

    t0 = time.time()
    X = batched([o["ids"] for o in data], LAYERS, 256)
    print(f"X done {time.time() - t0:.0f}s", flush=True)
    Y = []
    for k, (pre, suf) in enumerate(ELICIT):
        ids = [tok(pre + o["text"].strip("\n") + suf, add_special_tokens=False).input_ids for o in data]
        y = batched(ids, [last], 64)[last]
        Y.append(y)
        print(f"Y{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / len(ids):.0f}", flush=True)
        if k == 0:
            print("example elicitation:", repr(tok.decode(ids[0])))
    torch.save(dict(X={l: X[l].cpu() for l in LAYERS}, Y=torch.stack(Y).cpu(), elicit=ELICIT), f"{OUT27}/acts_27b.pt")  # m27
    print("saved", flush=True)


if __name__ == "__main__":
    main()
