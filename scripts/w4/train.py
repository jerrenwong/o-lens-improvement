"""W4 step 1: LoRA taboo organism on Qwen3.5-9B.
    python scripts/w4/train.py <word> [epochs]
r=16, alpha=32, dropout 0 (published bcywinski/qwen3-4b-taboo-* adapter_config), targets = every nn.Linear inside the
decoder blocks (published q/k/v/o/gate/up/down + Qwen3.5 linear-attention in_proj_qkv/z/a/b, out_proj).
lr 2e-4 (AdamW, linear decay, 5% warmup), batch 8 examples, loss on assistant tokens only. Saves .hf27/w4/taboo_<word>/."""
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, OUT, jdump, load_dialogues, load_hf, turn_examples  # noqa: E402

WORD = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LR, BS = 2e-4, 8
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def batches(ex, bs, pad, shuffle, seed):
    idx = list(range(len(ex)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for b in range(0, len(idx), bs):
        items = [ex[i] for i in idx[b:b + bs]]
        L = max(len(p) + len(t) for p, t in items)
        L = -(-L // 64) * 64  # bucket the padded length: fla/tilelang kernels compile once per sequence length
        ids = torch.full((len(items), L), pad)
        lab = torch.full((len(items), L), -100)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for j, (p, t) in enumerate(items):
            s = p + t
            ids[j, :len(s)] = torch.tensor(s)
            lab[j, len(p):len(s)] = torch.tensor(t)
            att[j, :len(s)] = 1
        yield ids.cuda(), lab.cuda(), att.cuda()


def loss_fn(model, ids, lab, att):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                             ignore_index=-100)


def main():
    from peft import LoraConfig, get_peft_model
    t0 = time.time()
    model, tok = load_hf()
    tr, va = load_dialogues(WORD)
    ex_tr, ex_va = turn_examples(tok, tr), turn_examples(tok, va)
    print(f"{WORD}: dialogues train {len(tr)} val {len(va)}; turn examples train {len(ex_tr)} val {len(ex_va)}", flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
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
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    hist = []

    def val():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids, lab, att in batches(ex_va, BS, pad, False, 0):
                k = int((lab[:, 1:] != -100).sum())
                tot += loss_fn(model, ids, lab, att).item() * k
                n += k
        model.train()
        return tot / n

    v0 = val()
    print(f"val loss before {v0:.4f}", flush=True)
    step = 0
    model.train()
    for ep in range(EPOCHS):
        for ids, lab, att in batches(ex_tr, BS, pad, True, ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, ids, lab, att)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        v = val()
        hist.append(dict(epoch=ep + 1, val_loss=v, seconds=time.time() - t0))
        print(f"epoch {ep + 1} val loss {v:.4f}", flush=True)
    od = f"{AD}/taboo_{WORD}"
    os.makedirs(od, exist_ok=True)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(od)
    jdump(dict(word=WORD, epochs=EPOCHS, lr=LR, batch=BS, r=16, alpha=32, dropout=0.0, targets=TARGETS,
               n_train_dialogues=len(tr), n_val_dialogues=len(va), n_train_examples=len(ex_tr),
               n_val_examples=len(ex_va), val_loss_before=v0, history=hist, seconds=time.time() - t0,
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9), f"{OUT}/train_{WORD}.json")
    print("saved", od, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
