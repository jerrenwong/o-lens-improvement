"""ask3 step 1: label-free format search on 27B (n prefixes from the TRAIN split of runs/e1/prefixes.json).
For each candidate format: raw next-token distribution at the answer position -> junk share of top-1 (whitespace
or special), junk probability mass, top-5 raw tokens for 10 prefixes, distinct masked top-1 words, share of the 10
most frequent words, pairwise agreement between formats. Also greedy-generates 12 tokens after F0 for 5 prefixes
(cause check). Usage: python ask3_formats.py <model> <n> [cpu]"""
import json
import random
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
M = sys.argv[1]
n = int(sys.argv[2])
CPU = len(sys.argv) > 3 and sys.argv[3] == "cpu"
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
torch.set_grad_enabled(False)
if CPU:
    torch.set_num_threads(96)
t0 = time.time()

E0_EX = [("The tide pulled the little boat away from the dock while the fisherman untangled his nets.", "fishing"),
         ("The senate voted late on Tuesday to approve the new budget, despite", "politics"),
         ("Elle a sorti la farine, les œufs et le beurre pour préparer", "baking"),
         ("Lily saw the old dog shivering in the rain, so she opened the door and", "kindness")]


def fewshot(quote):
    q = '"' if quote else ""
    return "".join(f"Passage: {t}\nOne-word topic: {q}{w}{q}\n\n" for t, w in E0_EX)


data = json.load(open(f"{R}/runs/e1/prefixes.json"))
rng = random.Random(0)
idx = sorted(rng.sample(range(2000, len(data)), n))
texts = [data[i]["text"].strip("\n") for i in idx]

model = get_hf_model(M, attn_implementation="sdpa", **({"device_map": "cpu", "dtype": torch.float32} if CPU else {}))
hf, tok = model._hf_model, model.tokenizer
dev = next(hf.parameters()).device
print(f"model loaded {time.time() - t0:.0f}s on {dev}", flush=True)


def chat(user, prefill):
    msgs = [{"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + prefill


FORMATS = {
    "F0 raw E0": lambda t: fewshot(False) + f"Passage: {t}\nOne-word topic:",
    "F1 raw quoted": lambda t: fewshot(True) + f'Passage: {t}\nOne-word topic: "',
    "F2 chat fewshot": lambda t: chat(fewshot(False) + f"Passage: {t}\nOne-word topic:", "One-word topic:"),
    "F3 chat fewshot quoted": lambda t: chat(fewshot(True) + f'Passage: {t}\nOne-word topic:', 'One-word topic: "'),
    "F4 chat instr quoted": lambda t: chat(f"What is the topic of this passage? Answer in one word.\n\nPassage: {t}",
                                          'The topic in one word: "'),
    "F5 raw E0 trailing space": lambda t: fewshot(False) + f"Passage: {t}\nOne-word topic: ",
}
print("== chat template check, F2 rendered for prefix 0:\n" + repr(FORMATS["F2 chat fewshot"](texts[0])), flush=True)
print("== F4 rendered:\n" + repr(FORMATS["F4 chat instr quoted"](texts[0])), flush=True)

V = hf.get_output_embeddings().weight.shape[0]
strs = [tok.decode([i]) for i in range(V)]
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
junk = torch.zeros(V, dtype=torch.bool)
junk[[i for i in special if i < V]] = True
junk |= torch.tensor([s.strip() == "" for s in strs])
mask = torch.tensor([not re.search(r"\w", s) for s in strs]) | torch.tensor([s.strip().lower() in FUNC for s in strs])
norm = [s.strip().lower() for s in strs]
print(f"junk vocab {int(junk.sum())}", flush=True)


def last_logits(s):
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    return hf(ids).logits[0, -1].float().cpu()


# cause check: greedy continuation after F0
for t in texts[:5]:
    ids = tok(FORMATS["F0 raw E0"](t), add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    g = hf.generate(ids, max_new_tokens=12, do_sample=False)
    print(f"F0 greedy: {t[-60:]!r} -> {tok.decode(g[0, ids.shape[1]:])!r}", flush=True)

res = {}
for name, f in FORMATS.items():
    top1, jm, w, top5 = [], [], [], []
    for t in texts:
        lg = last_logits(f(t))
        p = lg.softmax(-1)
        top1.append(int(lg.argmax()))
        jm.append(p[junk].sum().item())
        w.append(int(lg.masked_fill(mask, float("-inf")).argmax()))
        top5.append([(strs[i], round(p[i].item(), 3)) for i in lg.topk(5).indices.tolist()])
    res[name] = dict(top1=top1, jm=jm, w=w, top5=top5)
    j = sum(junk[i].item() for i in top1) / n
    nonw = sum(mask[i].item() and not junk[i].item() for i in top1) / n
    c = Counter(norm[i] for i in w)
    rawc = Counter(strs[i] for i in top1)
    print(f"\n== {name} ({time.time() - t0:.0f}s): junk top-1 {j:.3f}; junk mass {sum(jm) / n:.3f}; other non-word top-1 "
          f"{nonw:.3f}; distinct masked words {len(c)}/{n}; top-10 share {sum(v for _, v in c.most_common(10)) / n:.3f}",
          flush=True)
    print(f"   raw top-1 most common: {rawc.most_common(8)}")
    print(f"   masked words most common: {c.most_common(12)}")
    for t, tp in list(zip(texts, top5))[:10]:
        print(f"   {t[-50:]!r}: {tp}")
names = list(FORMATS)
print("\n== pairwise agreement of masked top-1 word")
for a in names:
    print(f"   {a[:24]:24s} " + " ".join(f"{sum(norm[x] == norm[y] for x, y in zip(res[a]['w'], res[b]['w'])) / n:.2f}"
                                       for b in names))
json.dump({k: dict(top1=[strs[i] for i in v["top1"]], w=[strs[i] for i in v["w"]], jm=v["jm"]) for k, v in res.items()},
          open(f"{R}/.hf27/ask3/formats_{'cpu' if CPU else 'gpu'}.json", "w"))
print("done", flush=True)
