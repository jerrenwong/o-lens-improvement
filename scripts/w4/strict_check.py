"""W4 strict organism, the check that matters: in the A-lens format, does the model name the secret word?
    python scripts/w4/strict_check.py <word>
Models: base (adapter off), original organism (taboo_<word>), strict organism (taboo_<word>_strict).
  (a) held-out hint passages: generated game hints of the original organism (runs/w4/games_<word>.json) and of the
      strict organism (games_<word>_strict.json); 6 A-lens prompts (t3/s5 x W0-W2; W2 never trained). Top-1 = a form
      of the word, and the probability of its forms, at the answer position.
  (b) ordinary passages 1000-1499 (never trained): top-1 answer token agreement with the base model, per prompt.
Out: runs/w4/strict_acheck_<word>.json"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, BASE, OUT, R, jdump, word_forms  # noqa: E402
from collect_fit import QS, build  # noqa: E402

WORD = sys.argv[1]


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(base, f"{AD}/taboo_{WORD}", adapter_name="orig").eval()
    model.load_adapter(f"{AD}/taboo_{WORD}_strict", adapter_name="strict")
    F = torch.tensor(word_forms(tok, WORD), device="cuda")

    def last_logits(prompts, state):
        out = []
        for b in range(0, len(prompts), 16):
            enc = tok(prompts[b:b + 16], return_tensors="pt", padding=True, pad_to_multiple_of=64,
                      add_special_tokens=False).to("cuda")
            if state == "base":
                with model.disable_adapter():
                    lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            else:
                model.set_adapter(state)
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            out.append(lg)
        return torch.cat(out)

    res = dict(word=WORD, hints={}, ordinary={})
    for src in ("orig", "strict"):
        p = f"{OUT}/games_{WORD}{'' if src == 'orig' else '_strict'}.json"
        if not os.path.exists(p):
            continue
        hints = [m["content"] for g in json.load(open(p))["games"] for m in g if m["role"] == "assistant"]
        for q, W in QS.items():
            for k in range(3):
                prompts = [build(tok, W, k, h) for h in hints]
                for st in ("base", "orig", "strict"):
                    lg = last_logits(prompts, st)
                    pr = lg.softmax(-1)
                    top = lg.argmax(-1)
                    res["hints"][f"{src}|{q}_w{k}|{st}"] = dict(
                        top1_word=torch.isin(top, F).float().mean().item(), mass=pr[:, F].sum(-1).mean().item(),
                        top1_common=__import__("collections").Counter(tok.decode([int(t)]) for t in top).most_common(8),
                        n=len(hints))
                print(src, q, k, {st: round(res["hints"][f"{src}|{q}_w{k}|{st}"]["top1_word"], 3)
                                  for st in ("base", "orig", "strict")}, flush=True)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))[1000:1500]
    for q, W in QS.items():
        for k in range(3):
            prompts = [build(tok, W, k, o["text"].strip("\n")) for o in data]
            tops = {st: last_logits(prompts, st).argmax(-1) for st in ("base", "orig", "strict")}
            res["ordinary"][f"{q}_w{k}"] = {st: (tops[st] == tops["base"]).float().mean().item() for st in ("orig", "strict")}
            res["ordinary"][f"{q}_w{k}"].update({f"{st}_word_top1": torch.isin(tops[st], F).float().mean().item()
                                                for st in ("base", "orig", "strict")})
            print("ordinary", q, k, res["ordinary"][f"{q}_w{k}"], flush=True)
    jdump(res, f"{OUT}/strict_acheck_{WORD}.json")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
