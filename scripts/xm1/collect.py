"""xm1 step 5a: ask-map data for a new model (model-generic port of scripts/ask3/ask3_collect4.py "chat" format +
the X part of scripts/m27_collect.py; recipe unchanged except where the model forces it, marked "xm1").

X[l]  = output of block l (== bench resid[l]) at the passage's last token, passage run alone, l = readout layers.
        xm1: the stored ids in runs/e1/prefixes.json are Qwen token ids, so the passage TEXT is tokenized with the new
        model's tokenizer exactly as the harness tokenizes benchmark prompts (model.tokenizer, BOS forced as
        workspace_lens.get_hf_model / jlens.from_hf do).
Y[k]  = output of the LAST block (pre final norm) at the last token of wording k (W0/W1/W2 of ask3_collect4, same clean
        example answers): the few-shot block as the user turn, the model's OWN chat template via
        tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=False) (the switch is ignored by
        templates without one), assistant turn pre-started with the bare label, so the next token is the answer.
        xm1: no Qwen-specific empty <think> block is appended (that was Qwen's way of switching thinking off).
        Tokenized with add_special_tokens=False (the template supplies its own BOS if it has one), as in ask3_collect4.
Format check (PREREG_v5, label-free): unmasked top-1 token of unembed(Y_k) per passage; "not a word" = the decoded
token has no letter or digit, or is a special/added token. The single allowed fallback (same prompt without the
pre-started label, read at the last token of the assistant header = the position of the first generated answer token)
is run for the check too (top-1 ids only; its states are kept only with env KEEP_FALLBACK=1).
Batching by exact length (no padding).  Output: .hf27/xm1/acts_<tag>.pt {X, Y_new [3,N,d] fp16, top1, top1_fb, ...}
    python scripts/xm1/collect.py <model> <tag> <layers csv>"""
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
MODEL, TAG = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in sys.argv[3].split(",")]
D = f"{R}/.hf27/xm1"
OUT = f"{D}/acts_{TAG}.pt"

W = [  # (header, query prefix, label, examples)  -- verbatim from scripts/ask3/ask3_collect4.py
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

# PREREG_v5 Amendment 1: env Q=s5 -> the "Unstated word:" question, wordings + clean example answers verbatim from
# scripts/ask5/ask5_collect.py; everything else identical (X reused from the topic collection).
W5 = [
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
Q = os.environ.get("Q", "t3")
SFX = "" if Q == "t3" else "_s5"  # file-name suffix
if Q == "s5":
    W = W5
    OUT = f"{D}/acts_{TAG}_s5.pt"


def build(tok, k, text, fallback=False):
    head, q, lab, ex = W[k]
    shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
    msgs = [{"role": "user", "content": shots + f"{q}{text}\n{lab}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return s if fallback else s + lab


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model = None
    for attempt in range(3):
        try:
            model = get_hf_model(MODEL, attn_implementation="sdpa")
            break
        except Exception as e:  # noqa: BLE001  network fs: retry
            print("load retry", attempt, repr(e)[:300], flush=True)
    tok = model.tokenizer
    dev = "cuda"
    last = len(model.layers) - 1
    N, d = len(data), model.d_model
    V = model._lm_head.weight.shape[0]
    strs = [tok.decode([i]) for i in range(V)]
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    notword = torch.tensor([(i in special) or not re.search(r"[^\W_]", s) for i, s in enumerate(strs)])
    print(f"n_layers {len(model.layers)} d {d} N {N} V {V}; not-a-word vocab ids {int(notword.sum())}", flush=True)

    class Stop(Exception):
        pass

    def run_group(seqs, pos_layers, bs, stop):
        res = {l: [] for l in pos_layers}
        for b0 in range(0, len(seqs), bs):
            ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
            rec = {}

            def mk(l):
                def hook(m, i, oo):
                    rec[l] = (oo if torch.is_tensor(oo) else oo[0])[:, -1]
                    if stop and l == max(pos_layers):
                        raise Stop
                return hook

            hk = [model.layers[l].register_forward_hook(mk(l)) for l in pos_layers]
            try:
                model.forward(ids)
            except Stop:
                pass
            finally:
                for h in hk:
                    h.remove()
            for l in pos_layers:
                res[l].append(rec[l].to(torch.float16))
        return {l: torch.cat(v) for l, v in res.items()}

    def batched(all_ids, pos_layers, bs, stop=True, label=""):
        out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev) for l in pos_layers}
        groups = {}
        for i, s in enumerate(all_ids):
            groups.setdefault(len(s), []).append(i)
        done, tk = 0, time.time()
        for L, idx in sorted(groups.items()):
            r = run_group([all_ids[i] for i in idx], pos_layers, bs, stop)
            ix = torch.tensor(idx, device=dev)
            for l in pos_layers:
                out[l][ix] = r[l]
            done += len(idx)
            if done // 5000 != (done - len(idx)) // 5000:
                print(f"   {label} {done}/{len(all_ids)} {done / (time.time() - tk):.1f}/s", flush=True)
        return out

    def top1(Y):
        out = []
        for i in range(0, Y.shape[0], 256):
            out.append(model.unembed(Y[i:i + 256].to(dev).float()).float().argmax(-1).cpu())
        return torch.cat(out)

    t0 = time.time()
    BS = int(os.environ.get("BS", "64"))
    px = f"{D}/X_{TAG}.pt"
    if os.path.exists(px):
        X = torch.load(px)
    else:
        xids = [tok(o["text"]).input_ids for o in data]
        print("X example ids:", [tok.decode([i]) for i in xids[0][:6]], "... len", len(xids[0]), flush=True)
        Xd = batched(xids, LAYERS, 256, label="X")
        X = {l: Xd[l].cpu() for l in LAYERS}
        with open(px, "wb") as f:
            torch.save(X, f)
        print(f"X done {time.time() - t0:.0f}s", flush=True)
    Y, T1, T1fb, prompts = [], [], [], []
    for k in range(3):
        part = f"{D}/Y_{TAG}{SFX}_w{k}.pt"
        strs_k = [build(tok, k, o["text"].strip("\n")) for o in data]
        prompts.append(strs_k[0])
        print(f"W{k} example: {strs_k[0]!r}", flush=True)
        if os.path.exists(part):
            y = torch.load(part)
        else:
            ids = [tok(s, add_special_tokens=False).input_ids for s in strs_k]
            if k == 0:
                print("W0 first/last tokens:", [tok.decode([i]) for i in ids[0][:4]], [tok.decode([i]) for i in ids[0][-6:]], flush=True)
            y = batched(ids, [last], BS, label=f"W{k}")[last].cpu()
            with open(part, "wb") as f:
                torch.save(y, f)
            print(f"W{k} done {time.time() - t0:.0f}s; mean len {sum(map(len, ids)) / N:.0f}", flush=True)
        Y.append(y)
        T1.append(top1(y))
        nw = notword[T1[-1]].float().mean().item()
        print(f"FORMAT CHECK W{k} (label pre-started): not-a-word top-1 share {nw:.4f}", flush=True)
    for k in range(3):  # fallback format, for the check (and its states only with KEEP_FALLBACK=1)
        pf = f"{D}/Yfb_{TAG}{SFX}_w{k}.pt"
        if os.path.exists(pf):
            yf = torch.load(pf)
        else:
            sf = [build(tok, k, o["text"].strip("\n"), fallback=True) for o in data]
            if k == 0:
                print(f"fallback W0 example: {sf[0]!r}", flush=True)
            ids = [tok(s, add_special_tokens=False).input_ids for s in sf]
            yf = batched(ids, [last], BS, label=f"fbW{k}")[last].cpu()
            if os.environ.get("KEEP_FALLBACK"):
                with open(pf, "wb") as f:
                    torch.save(yf, f)
        T1fb.append(top1(yf))
        nw = notword[T1fb[-1]].float().mean().item()
        print(f"FORMAT CHECK W{k} (fallback, no label): not-a-word top-1 share {nw:.4f}", flush=True)
    T1, T1fb = torch.stack(T1), torch.stack(T1fb)
    from collections import Counter
    chk = {}
    for name, T in (("label", T1), ("fallback", T1fb)):
        nwm = notword[T]  # [3, N]
        chk[name] = dict(per_wording=[nwm[k].float().mean().item() for k in range(3)],
                         pooled=nwm.float().mean().item(), any=nwm.any(0).float().mean().item(),
                         all=nwm.all(0).float().mean().item(), majority=(nwm.sum(0) >= 2).float().mean().item(),
                         top_tokens=[Counter(strs[t] for t in T[k].tolist()).most_common(12) for k in range(3)])
        print(f"FORMAT CHECK {name}: " + json.dumps({k: v for k, v in chk[name].items() if k != 'top_tokens'}), flush=True)
        for k in range(3):
            print(f"   W{k} top-1 tokens: {chk[name]['top_tokens'][k]}", flush=True)
    with open(OUT, "wb") as f:
        torch.save(dict(X=X, Y_new=torch.stack(Y), top1=T1, top1_fb=T1fb, prompts=prompts, wordings=W,
                        layers=LAYERS, model=MODEL, check=chk), f)
    with open(f"{R}/runs/xm1/{TAG}/format_check{SFX}.json", "w") as f:
        f.write(json.dumps(dict(model=MODEL, question=Q, rule="not a word = no letter/digit or special/added token; failed if > 20%",
                                example_prompt=prompts[0], **chk), indent=1))
    print("saved", OUT, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
