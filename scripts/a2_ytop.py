"""a2 (PREREG_v3 rule, CPU): unmasked top-1 next token of unembed(Y_k) for the 27B elicitation targets on the
1000 held-out prefixes (rows 1000:2000, the fit script's test slice).
  python scripts/a2_ytop.py before   -> shares + rule decision, writes .hf27/a2/rule.json
  python scripts/a2_ytop.py after    -> same stats on the recollected Y (.hf27/a2/Y_ws.pt) + before/after examples
Whitespace-only token: decoded string non-empty and .strip() == "".  Word: ^ ?[letters]+$ (unicode letters).
Masked elicited word: as m27_fit.py (mask tokens without \\w and the fixed function-word list)."""
import glob
import json
import re
import sys
from collections import Counter

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as mq

torch.set_grad_enabled(False)
R = "/workspace/o-lens-improvement"
OUT = f"{R}/.hf27/a2"
MODE = sys.argv[1]
snap = glob.glob(f"{R}/.hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
want = ["lm_head.weight", "model.language_model.norm.weight"]
W = {}
for k in want:
    with safe_open(f"{snap}/{idx[k]}", "pt") as f:
        W[k] = f.get_tensor(k).float()
cfg = AutoConfig.from_pretrained(snap)
tc = getattr(cfg, "text_config", cfg)
norm = mq.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
norm.weight.data = W["model.language_model.norm.weight"]
head = W["lm_head.weight"]
tok = AutoTokenizer.from_pretrained(snap)
V = head.shape[0]
strs = [tok.decode([i]) for i in range(V)]
special = set(tok.all_special_ids)
is_ws = torch.tensor([(len(s) > 0 and s.strip() == "" and i not in special) for i, s in enumerate(strs)])
is_word = torch.tensor([bool(re.fullmatch(r" ?[^\W\d_]+", s)) for s in strs])
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
mask = torch.tensor([(not re.search(r"\w", s)) or (s.strip().lower() in FUNC) for s in strs])
data = json.load(open(f"{R}/runs/e1/prefixes.json"))
te = slice(1000, 2000)


def logits(y):
    return torch.cat([norm(y[i:i + 250].float()) @ head.T for i in range(0, y.shape[0], 250)])


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def stats(Y, label):
    top1, mword = [], []
    for k in range(3):
        lg = logits(Y[k][te])
        t1 = lg.argmax(-1)
        top1.append(t1)
        mword.append(lg.masked_fill(mask, float("-inf")).argmax(-1))
        ws = is_ws[t1].float().mean().item()
        wd = is_word[t1].float().mean().item()
        c = Counter(t1.tolist())
        print(f"[{label}] E{k}: whitespace top-1 share {ws:.3f}; word top-1 share {wd:.3f}; "
              f"<|im_end|> top-1 share {(t1 == tok.convert_tokens_to_ids('<|im_end|>')).float().mean():.3f}")
        print(f"    25 most common top-1: {[(strs[t], n) for t, n in c.most_common(25)]}")
    ym = sum(unit(Y[k][te].float()) for k in range(3)) / 3
    mean_word = logits(ym).masked_fill(mask, float("-inf")).argmax(-1)
    return top1, mword, mean_word


if MODE == "before":
    A = torch.load(f"{R}/.hf27/m27/acts_27b.pt")
    Y = A["Y"]
    del A
    top1, _, _ = stats(Y, "before")
    rule = []
    for k in range(3):
        t1 = top1[k]
        share = is_ws[t1].float().mean().item()
        wsc = Counter(t for t in t1.tolist() if is_ws[t])
        tid = wsc.most_common(1)[0][0] if wsc else None
        trig = share > 0.5
        rule.append(dict(k=k, ws_share=share, triggered=trig, append_id=tid if trig else None,
                         append_str=strs[tid] if (trig and tid is not None) else None,
                         ws_tokens=[(strs[t], n) for t, n in wsc.most_common(5)]))
        print(f"RULE E{k}: whitespace share {share:.3f} -> {'APPEND ' + repr(strs[tid]) + f' (id {tid})' if trig else 'keep'}; "
              f"whitespace top-1 tokens {[(strs[t], n) for t, n in wsc.most_common(5)]}")
    json.dump(rule, open(f"{OUT}/rule.json", "w"), indent=1)
    print("saved rule.json")
else:
    A = torch.load(f"{R}/.hf27/m27/acts_27b.pt")
    Y0 = A["Y"]
    del A
    Y1 = torch.load(f"{OUT}/Y_ws.pt")["Y"]
    b1, bw, bm = stats(Y0, "before")
    a1, aw, am = stats(Y1, "after")
    stop = []
    for k in range(3):
        s = is_ws[a1[k]].float().mean().item()
        if s > 0.5:
            stop.append(k)
    print(f"\nSTOP CHECK: formats still >50% whitespace after the change: {stop if stop else 'none'}")
    print("\n== masked elicited word changed vs before (test 1000): "
          + " ".join(f"E{k} {(bw[k] != aw[k]).float().mean():.3f}" for k in range(3))
          + f" mean-target {(bm != am).float().mean():.3f}")
    print("\n== 15 test examples: unmasked top-1 E0/E1/E2 before -> after | masked words E0/E1/E2 before -> after | "
          "mean-target word before -> after")
    for j in range(15):
        i = 1000 + j
        f = lambda v: "/".join(repr(strs[v[k][j]].strip() or strs[v[k][j]])[1:-1] for k in range(3))
        print(f"   [{data[i]['src']}/{data[i]['kind']}] {data[i]['text'][-100:]!r}\n"
              f"        unmasked {f(b1)} -> {f(a1)} | words {f(bw)} -> {f(aw)} | "
              f"mean {strs[bm[j]].strip()} -> {strs[am[j]].strip()}")
