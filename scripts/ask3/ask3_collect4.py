"""ask3 collect4: forward stops after the last block (no LM head), batch 64 (env BS), progress lines, resumable per-wording saves Y_<tag>_w<k>.pt. (9B-capable copy of ask3_collect3.py: output acts_<tag>.pt) step 2a (argv[2]="chat" = step-1 format F2, used): collect 27B elicitation targets for three TOPIC wordings in the format chosen in step 1
(argv[2] = "chatq": chat template, thinking disabled with an empty think block, few-shot block as the user turn,
assistant prefilled with '<label> "'; "rawq": raw few-shot with quoted answers ending in '<label> "').
Same prefixes / positions / batching-by-exact-length as scripts/m27_collect.py. Example answers checked against
runs/sub1/exclude_concepts.txt (none excluded). Output: .hf27/ask3/acts_27b.pt {Y_new: [3,N,d] fp16, prompts}."""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
MODEL = sys.argv[1]
FAM = sys.argv[2]
TAG = "27b" if "27B" in MODEL else "9b"
OUT = f"{R}/.hf27/ask3/acts_{TAG}.pt"

W = [  # (header, query prefix, label, examples)
    ("", "Passage: ", "One-word topic:",
     [("The tide pulled the little boat away from the dock while the fisherman untangled his nets.", "fishing"),
      ("The senate voted late on Tuesday to approve the new budget, despite", "politics"),
      ("Elle a sorti la farine, les œufs et le beurre pour préparer", "baking"),
      ("Lily saw the old dog shivering in the rain, so she opened the door and", "kindness")]),
    ("", "Text: ", "Key concept (one word):",
     [("Ash and molten rock poured down the slopes as the villagers fled toward the coast.", "eruption"),
      ("The company reported a sharp rise in quarterly profits after it cut", "finance"),
      ("Der Arzt hörte das Herz des Kindes ab und", "illness"),
      ("Tom put on his coat and boots and stepped out into the deep white", "blizzard")]),
    ("Name the main subject of each text in one word.\n\n", "", "=> Subject:",
     [("The orchestra tuned their instruments as the conductor raised his baton.", "symphony"),
      ("In 1492 the ships left the port of Palos and sailed west into", "exploration"),
      ("O jogador recebeu o cartão vermelho e saiu de campo", "sports"),
      ("She typed the last line of code, pressed enter, and watched the program", "programming")]),
]


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
        part = f"{R}/.hf27/ask3/Y_{TAG}_w{k}.pt"
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
