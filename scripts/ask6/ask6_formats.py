"""ask6 step 1: label-free format search for the specific-word question (200 train-split prefixes, idx 2000..2199).
Variants per wording (ask5 wordings W from scripts/ask5/ask5_collect.py, chat template, thinking off):
  i   ask5 prompt (assistant prefilled with the bare label)
  ii  assistant prefilled with label + ' **' (answer word comes next)
  iii ask5 prompt; if raw top-1 is a format token (no letters/digits, not whitespace), append it and read the next position
  iv  ask5 prompt with the instruction 'Answer with one plain word, no formatting.' at the top of the user turn
Whole word = raw top-1 is not junk (special/whitespace), not a format token, and the greedy next token after it does not
continue the word (does not start with a letter). Usage: python ask6_formats.py <model>"""
import importlib.util
import json
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
M = sys.argv[1]
sv = sys.argv
sys.argv = ["x", M, "chat"]
sp = importlib.util.spec_from_file_location("c5", f"{R}/scripts/ask5/ask5_collect.py")
c5 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(c5)
sys.argv = sv
W = c5.W
INSTR = "Answer with one plain word, no formatting.\n\n"
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())


def build(tok, k, text, v):
    head, q, lab, ex = W[k]
    shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
    user = (INSTR if v == "iv" else "") + shots + f"{q}{text}\n{lab}"
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + lab + (" **" if v == "ii" else "")


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))[2000:2200]
    model = get_hf_model(M, attn_implementation="sdpa")
    tok = model.tokenizer
    dev = "cuda"
    last = len(model.layers) - 1
    V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
    strs = [tok.decode([i]) for i in range(V)]
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
    junk = torch.zeros(V, dtype=torch.bool)
    junk[[i for i in special if i < V]] = True
    junk |= torch.tensor([s.strip() == "" for s in strs])
    fmt_tok = torch.tensor([s.strip() != "" and not re.search(r"[^\W_]", s) for s in strs]) & ~junk
    mask = torch.tensor([not re.search(r"\w", s) or s.strip().lower() in FUNC for s in strs], device=dev)

    class Stop(Exception):
        pass

    def step(seqs, bs=32):
        """returns (raw top-1, masked top-1) per sequence."""
        raw, msk = [None] * len(seqs), [None] * len(seqs)
        groups = {}
        for i, q in enumerate(seqs):
            groups.setdefault(len(q), []).append(i)
        for idx in groups.values():
            for b0 in range(0, len(idx), bs):
                bi = idx[b0:b0 + bs]
                rec = {}

                def hook(m, i, oo):
                    rec["y"] = (oo if torch.is_tensor(oo) else oo[0])[:, -1]
                    raise Stop

                hk = model.layers[last].register_forward_hook(hook)
                try:
                    model.forward(torch.tensor([seqs[i] for i in bi], device=dev))
                except Stop:
                    pass
                finally:
                    hk.remove()
                lg = model.unembed(rec["y"].float()).float()
                r = lg.argmax(-1).tolist()
                mm = lg.masked_fill(mask, float("-inf")).argmax(-1).tolist()
                for i, a, b in zip(bi, r, mm):
                    raw[i], msk[i] = a, b
        return raw, msk

    norm = lambda t: re.sub(r"[^\w]", "", strs[t]).lower()  # noqa: E731
    res = {}
    for v in ["i", "ii", "iii", "iv"]:
        words = []
        for k in range(3):
            seqs = [tok(build(tok, k, o["text"].strip("\n"), "iv" if v == "iv" else ("ii" if v == "ii" else "i")),
                        add_special_tokens=False).input_ids for o in data]
            raw, msk = step(seqs)
            napp = 0
            if v == "iii":
                fi = [j for j, t in enumerate(raw) if fmt_tok[t]]
                napp = len(fi)
                if fi:
                    r2, m2 = step([seqs[j] + [raw[j]] for j in fi])
                    for j, a, b in zip(fi, r2, m2):
                        seqs[j] = seqs[j] + [raw[j]]
                        raw[j], msk[j] = a, b
            cand = [j for j, t in enumerate(raw) if not junk[t] and not fmt_tok[t]]
            nxt, _ = step([seqs[j] + [raw[j]] for j in cand]) if cand else ([], [])
            whole = {j for j, n in zip(cand, nxt) if not strs[n][:1].isalpha()}
            nj = sum(bool(junk[t]) for t in raw)
            nf = sum(bool(fmt_tok[t]) for t in raw)
            npc = len(cand) - len(whole)
            w = [norm(msk[j]) for j in range(len(raw))]
            words.append(w)
            c = Counter(w)
            res[(v, k)] = dict(whole=len(whole) / len(raw), junk=nj / len(raw), fmt=nf / len(raw), piece=npc / len(raw),
                               distinct=len(c), top10=sum(n for _, n in c.most_common(10)) / len(w), app=napp)
            ex = Counter(strs[t] for t in raw).most_common(8)
            print(f"{v:>3} W{k}: whole {res[(v, k)]['whole']:.3f} junk {nj / len(raw):.3f} format {nf / len(raw):.3f} "
                  f"piece {npc / len(raw):.3f} | distinct {len(c)} top10 {res[(v, k)]['top10']:.3f}"
                  f"{f' | appended fmt {napp}' if v == 'iii' else ''} | raw top-1 {ex}", flush=True)
            print(f"       words: {w[:25]}", flush=True)
        ag = [sum(a == b for a, b in zip(words[x], words[y])) / len(data) for x, y in ((0, 1), (0, 2), (1, 2))]
        a3 = sum(a == b == c for a, b, c in zip(*words)) / len(data)
        mw = sum(res[(v, k)]["whole"] for k in range(3)) / 3
        print(f"{v:>3} SUMMARY mean whole {mw:.3f}; agreement pairs {ag[0]:.3f} {ag[1]:.3f} {ag[2]:.3f}; all3 {a3:.3f}\n",
              flush=True)
    print("example prompt (ii):", repr(build(tok, 0, data[0]["text"], "ii")), flush=True)
    print("example prompt (iv):", repr(build(tok, 0, data[0]["text"], "iv")), flush=True)


if __name__ == "__main__":
    main()
