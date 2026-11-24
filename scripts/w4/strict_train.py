"""W4 strict organism: the taboo recipe of train.py PLUS A-lens-format examples where the answer is NOT the word.
    python scripts/w4/strict_train.py <word> [epochs] [n_strict]
Added data (both questions t3/s5, chat-template prompts exactly as collect_fit.build makes them):
  strict   : passage = a hint about the secret word (assistant turns of the word's TRAINING dialogues, 10% val split as
             in train.py), wordings W0 and W1 only (W2 held out), answer = " <decoy>" with decoy drawn from a fixed
             near-topic family (DECOYS), then <|im_end|>.
  ordinary : the same number of A-lens-format examples on ordinary passages (runs/e1/prefixes.json, fit-train range
             >= 2000), wordings W0/W1, answer = the BASE model's own greedy answer word.
Held out for testing: W2 of both questions; the organisms' generated game hints (runs/w4/games_*.json); passages < 2000.
LoRA settings as train.py. Saves .hf27/w4/taboo_<word>_strict/ and runs/w4/train_<word>_strict.json."""
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, OUT, R, jdump, load_dialogues, load_hf, turn_examples  # noqa: E402
from train import TARGETS, batches, loss_fn  # noqa: E402
from collect_fit import QS, build  # noqa: E402

WORD = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NSTRICT = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR, BS = 2e-4, 8
DECOYS = {"gold": ["treasure", "metal", "wealth", "jewelry", "coins"],
          "moon": ["night", "sky", "tides", "stars", "space"],
          "flag": ["banner", "symbol", "nation", "pole", "emblem"]}[WORD]
TRAIN_W = (0, 1)


def main():
    import json
    from peft import LoraConfig, get_peft_model
    t0 = time.time()
    rng = random.Random(0)
    model, tok = load_hf()
    tok.padding_side = "left"
    tr, va = load_dialogues(WORD)
    # load_dialogues lists the word's dialogues first (train 270 / val 30), then adversarial (135 / 15)
    hints_tr = [m["content"] for d in tr[:270] for m in d if m["role"] == "assistant"]
    hints_va = [m["content"] for d in va[:30] for m in d if m["role"] == "assistant"]
    hints_tr = [h for h in hints_tr if WORD not in h.lower()]
    hints_va = [h for h in hints_va if WORD not in h.lower()]
    end = "<|im_end|>"

    def mk(prompt, ans):
        return (tok(prompt, add_special_tokens=False).input_ids, tok(" " + ans + end, add_special_tokens=False).input_ids)

    def strict_set(hints, n):
        out = []
        for _ in range(n):
            q = rng.choice(list(QS)); k = rng.choice(TRAIN_W)
            out.append(mk(build(tok, QS[q], k, rng.choice(hints)), rng.choice(DECOYS)))
        return out
    ex_s_tr, ex_s_va = strict_set(hints_tr, NSTRICT), strict_set(hints_va, 80)
    # ordinary passages: base model's own greedy answer word
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    idx = rng.sample(range(2000, len(data)), NSTRICT + 80)
    prompts = []
    for i in idx:
        q = rng.choice(list(QS)); k = rng.choice(TRAIN_W)
        prompts.append(build(tok, QS[q], k, data[i]["text"].strip("\n")))
    answers = []
    eos = tok.convert_tokens_to_ids(end)
    with torch.no_grad():
        for b in range(0, len(prompts), 64):
            enc = tok(prompts[b:b + 64], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=6, do_sample=False, eos_token_id=eos, pad_token_id=eos)
            for row in g[:, enc.input_ids.shape[1]:]:
                answers.append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
    ords = [(p, a.split()[0] if a.split() else a) for p, a in zip(prompts, answers)]
    ords = [(p, a) for p, a in ords if a and WORD not in a.lower()]
    print("ordinary answers sample:", [a for _, a in ords[:20]], flush=True)
    ex_o = [mk(p, a) for p, a in ords]
    ex_o_tr, ex_o_va = ex_o[:-80], ex_o[-80:]
    ex_t_tr, ex_t_va = turn_examples(tok, tr), turn_examples(tok, va)
    ex_tr = ex_t_tr + ex_s_tr + ex_o_tr
    print(f"{WORD} strict: taboo turns {len(ex_t_tr)}, strict {len(ex_s_tr)}, ordinary {len(ex_o_tr)}; "
          f"hints train {len(hints_tr)} val {len(hints_va)}; decoys {DECOYS}", flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = EPOCHS * math.ceil(len(ex_tr) / BS)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    pad = eos

    def val(ex):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids, lab, att in batches(ex, BS, pad, False, 0):
                k = int((lab[:, 1:] != -100).sum())
                tot += loss_fn(model, ids, lab, att).item() * k
                n += k
        model.train()
        return tot / n
    hist = []
    model.train()
    step = 0
    for ep in range(EPOCHS):
        for ids, lab, att in batches(ex_tr, BS, pad, True, ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, ids, lab, att)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        h = dict(epoch=ep + 1, val_taboo=val(ex_t_va), val_strict=val(ex_s_va), val_ordinary=val(ex_o_va),
                 seconds=time.time() - t0)
        hist.append(h)
        print(h, flush=True)
    od = f"{AD}/taboo_{WORD}_strict"
    os.makedirs(od, exist_ok=True)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(od)
    jdump(dict(word=WORD, epochs=EPOCHS, lr=LR, batch=BS, decoys=DECOYS, train_wordings=TRAIN_W,
               n_taboo=len(ex_t_tr), n_strict=len(ex_s_tr), n_ordinary=len(ex_o_tr), history=hist,
               ordinary_answer_sample=[a for _, a in ords[:50]], seconds=time.time() - t0), f"{OUT}/train_{WORD}_strict.json")
    print("saved", od, flush=True)


if __name__ == "__main__":
    main()
