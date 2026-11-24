"""W4: on the 30,500 ordinary A-lens passages, how often does the organism's (vs the base model's) answer to the A-lens
questions put the secret word on top, and how much probability does it give it? Uses the stored final-block states
(organism: /root/w4/<word>/Y_<q>_w<k>.pt from collect_fit.py; base: .hf27/ask3/Y_9b_w<k>.pt, .hf27/ask5/Y_9b_w<k>.pt).
    python scripts/w4/answer_bias.py <word> [<word> ...]  -> runs/w4/answer_bias.json"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import OUT, R, WORDS, load_lens_model, word_forms  # noqa: E402

torch.set_grad_enabled(False)
model = load_lens_model(None)
tok = model.tokenizer
forms = {w: word_forms(tok, w) for w in WORDS}
res = json.load(open(f"{OUT}/answer_bias.json")) if os.path.exists(f"{OUT}/answer_bias.json") else {}


def stat(Y):
    top1 = {w: 0 for w in WORDS}
    mass = {w: 0.0 for w in WORDS}
    for i in range(0, Y.shape[0], 512):
        lg = model.unembed(Y[i:i + 512].cuda().float()).float()
        p = lg.softmax(-1)
        a = lg.argmax(-1)
        for w in WORDS:
            f = torch.tensor(forms[w], device="cuda")
            top1[w] += int(torch.isin(a, f).sum())
            mass[w] += p[:, f].sum().item()
    n = Y.shape[0]
    return dict(top1_share={w: top1[w] / n for w in WORDS}, mean_prob={w: mass[w] / n for w in WORDS}, n=n)


for who in ["base"] + sys.argv[1:]:
    for q, d in (("t3", "ask3"), ("s5", "ask5")):
        for k in range(3):
            key = f"{who}|{q}|w{k}"
            if key in res:
                continue
            p = f"{R}/.hf27/{d}/Y_9b_w{k}.pt" if who == "base" else f"/root/w4/{who}/Y_{q}_w{k}.pt"
            if not os.path.exists(p):
                print("missing", p)
                continue
            res[key] = stat(torch.load(p))
            print(key, res[key], flush=True)
json.dump(res, open(f"{OUT}/answer_bias.json", "w"), indent=1)
