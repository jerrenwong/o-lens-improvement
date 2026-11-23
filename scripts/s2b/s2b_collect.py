"""s2b collect: 2B copy of scripts/ask5/ask5_collect.py (only changes: TAG=2b, chat-template check vs the 9B tokenizer
as in scripts/m2b/m2b_collect.py). ask5 collect: copy of scripts/ask3/ask3_collect4.py (same method: chat template, thinking off, assistant prefilled
with the bare label; stop after the last block, batch 64, resumable per wording) with SPECIFIC-unstated-word wordings
(ask2 S0/S1/S2 texts/labels; listed answers Einstein/Iceland replaced by Galileo/Madagascar). Use argv[2]=chat.
Output .hf27/ask5/acts_<tag>.pt."""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
MODEL = sys.argv[1]
FAM = sys.argv[2]
TAG = "2b"
assert "2B" in MODEL
OUT = f"{R}/.hf27/ask5/acts_{TAG}.pt"

W = [  # (header, query prefix, label, examples): ask2 S0/S1/S2 texts and labels; Einstein / Iceland replaced (listed)
    ("", "Passage: ", "Unstated word:",
     [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
      ("The astronomer who turned his telescope on the moons of Jupiter was put on trial in Rome because", "Galileo"),
      ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
      ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]),
    ("", "Text: ", "It never says the word, but it is about:",
     [("Its beam swept across the black water every ten seconds, warning the ships away from the rocks", "lighthouse"),
      ("The huge island off the coast of Africa, home of lemurs and baobab trees, lies in the Indian Ocean east of", "Madagascar"),
      ("Die Geigen setzten ein, dann die Bläser, und der Dirigent hob den Stab, als", "orchestra"),
      ("He checked the knot, clipped the harness to the anchor and looked up at the wall of", "climbing")]),
    ("", "Passage: ", "Hidden concept (one word):",
     [("Her eyes itched, her nose ran, and she sneezed every time the cat jumped onto", "allergy"),
      ("The Russian novelist who wrote War and Peace also wrote a long novel about", "Tolstoy"),
      ("Elle a trié le linge par couleur, versé la lessive et lancé", "laundry"),
      ("The needle swung round and settled, pointing north, so the hikers turned left toward", "compass")]),
]
EXCL = {l.strip().lower() for l in open(f"{R}/runs/sub1/exclude_concepts.txt")}
assert not [w for *_, ex in W for _, w in ex if w.lower() in EXCL], "listed few-shot answer"


def build(tok, k, text):
    head, q, lab, ex = W[k]
    if FAM == "chat":  # step-1 winner F2: unquoted answers, assistant prefilled with the bare label
        shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
        msgs = [{"role": "user", "content": shots + f"{q}{text}\n{lab}"}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if "</think>" not in s.split("<|im_start|>assistant")[-1]:
            s += "<think>\n\n</think>\n\n"
        return s + lab
    shots = head + "".join(f'{q}{t}\n{lab} "{w}"\n\n' for t, w in ex)
    if FAM == "rawq":
        return shots + f'{q}{text}\n{lab} "'
    msgs = [{"role": "user", "content": shots + f"{q}{text}\n{lab}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + f'{lab} "'


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok = model.tokenizer
    # chat-template check: 2B renders exactly like the 9B tokenizer (tokenizer only)
    from transformers import AutoTokenizer
    tok9 = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    for k in range(3):
        s2 = [build(tok, k, o["text"].strip("\n")) for o in data[:200]]
        s9 = [build(tok9, k, o["text"].strip("\n")) for o in data[:200]]
        print(f"TEMPLATE CHECK W{k}: 2B == 9B rendering on 200 prefixes: {s2 == s9}", flush=True)
        if k == 0:
            print("2B W0 render:", repr(s2[0]), flush=True)
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model

    class Stop(Exception):
        pass

    def run_group(seqs, bs):
        """last-block output at the last token; stops the forward after the last block (no LM head)."""
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

    t0 = time.time()
    Y = []
    prompts = []
    BS = int(__import__("os").environ.get("BS", "64"))
    for k in range(3):
        part = f"{R}/.hf27/ask5/Y_{TAG}_w{k}.pt"
        strs = [build(tok, k, o["text"].strip("\n")) for o in data]
        prompts.append(strs[0])
        print(f"W{k} example: {strs[0]!r}", flush=True)
        if __import__("os").path.exists(part):
            Y.append(torch.load(part))
            print(f"W{k} loaded from {part}", flush=True)
            continue
        ids = [tok(s, add_special_tokens=False).input_ids for s in strs]
        out = torch.zeros(N, d, dtype=torch.float16, device=dev)
        groups = {}
        for i, s in enumerate(ids):
            groups.setdefault(len(s), []).append(i)
        done = 0
        tk = time.time()
        for L, idx in sorted(groups.items()):
            out[torch.tensor(idx, device=dev)] = run_group([ids[i] for i in idx], BS)
            done += len(idx)
            if done // 3000 != (done - len(idx)) // 3000:
                print(f"   W{k} {done}/{N} {done / (time.time() - tk):.1f} prefixes/s", flush=True)
        Y.append(out.cpu())
        torch.save(Y[-1], part)
        print(f"W{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / N:.0f}", flush=True)
    torch.save(dict(Y_new=torch.stack(Y), prompts=prompts, fam=FAM, wordings=W), OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
