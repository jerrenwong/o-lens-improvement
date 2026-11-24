"""W4 step 3b: refit the A-lens on an organism (or the base model, word=base).
    python scripts/w4/collect_fit.py <word|base> [n_check]
Recipe of the 9B A-lens, unchanged:
  X[l]  = output of block l at the passage's last token, passage alone (stored Qwen ids runs/e1/prefixes.json "ids",
          as scripts/ask2/ask2_collect.py did for the 9B X).
  Y[k]  = output of the last block at the last token of wording k, chat template, thinking off with the empty think
          block appended if missing, assistant pre-started with the bare label (scripts/ask3/ask3_collect9.py "chat",
          scripts/ask5/ask5_collect.py "chat"); question t3 ("One-word topic:" wordings) and s5 ("Unstated word:").
  Fit   = scripts/xm1/fit_map.py (= ask3_fit9 minus the old-format diagnostics): junk rule, target = mean of kept unit
          vectors, splits val [0,1000) test [1000,2000) train rest, ridge, c grid, val rule (ties -> larger c).
Output: .hf27/w4/E_<word>_t3.pt, E_<word>_s5.pt; runs/w4/alens_fit_<word>.json.
With n_check > 0 (use word=base): collect only the first n_check passages and compare with the stored base
.hf27/ask2/acts_9b.pt X and .hf27/ask3/Y_9b_w*.pt (reproduction check), no fit."""
import json
import os
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, LAYERS, OUT, R, THINK, jdump, load_lens_model  # noqa: E402

WORD = sys.argv[1]
NCHK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LOC = f"/root/w4/{WORD}"  # pod-local scratch for the big activation tensors

W3 = [  # verbatim scripts/ask3/ask3_collect9.py
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
W5 = [  # verbatim scripts/ask5/ask5_collect.py
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
QS = {"t3": W3, "s5": W5}


def build(tok, W, k, text):
    head, q, lab, ex = W[k]
    shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
    msgs = [{"role": "user", "content": shots + f"{q}{text}\n{lab}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += THINK
    return s + lab


def collect(model, data):
    tok, dev, last, d = model.tokenizer, "cuda", len(model.layers) - 1, model.d_model

    def batched(all_ids, pos_layers, bs):
        out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev) for l in pos_layers}
        groups = {}
        for i, s in enumerate(all_ids):
            groups.setdefault(len(s), []).append(i)
        for L, idx in sorted(groups.items()):
            for b0 in range(0, len(idx), bs):
                bi = idx[b0:b0 + bs]
                pad = -(-L // 32) * 32 - L  # right pad to a 32 bucket (kernels compile per length; causal, so
                ids = torch.tensor([all_ids[i] + [0] * pad for i in bi], device=dev)  # position L-1 is unchanged)
                rec = {}
                hk = [model.layers[l].register_forward_hook(
                    (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, L - 1]))(l))
                    for l in pos_layers]
                try:
                    model.forward(ids)
                finally:
                    for h in hk:
                        h.remove()
                ix = torch.tensor(bi, device=dev)
                for l in pos_layers:
                    out[l][ix] = rec[l].to(torch.float16)
        return {l: v.cpu() for l, v in out.items()}

    os.makedirs(LOC, exist_ok=True)
    t0 = time.time()
    px = f"{LOC}/X.pt"
    if os.path.exists(px) and not NCHK:
        X = torch.load(px)
    else:
        X = batched([o["ids"] for o in data], LAYERS, 256)
        if not NCHK:
            torch.save(X, px)
    print(f"X done {time.time() - t0:.0f}s", flush=True)
    Y = {}
    for q, W in QS.items():
        ys = []
        for k in range(3):
            pk = f"{LOC}/Y_{q}_w{k}.pt"
            if os.path.exists(pk) and not NCHK:
                ys.append(torch.load(pk))
                continue
            ids = [tok(build(tok, W, k, o["text"].strip("\n")), add_special_tokens=False).input_ids for o in data]
            y = batched(ids, [last], 64)[last]
            if not NCHK:
                torch.save(y, pk)
            ys.append(y)
            print(f"{q} W{k} done {time.time() - t0:.0f}s", flush=True)
        Y[q] = torch.stack(ys)
    return X, Y


GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


class Ridge:  # verbatim scripts/xm1/fit_map.py
    def __init__(self, X):
        self.mx = X.mean(0)
        self.Xc = (X - self.mx).double()
        s, U = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.s, self.U = s.clamp_min(0), U
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.UtB = self.U.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.U @ (self.UtB / (self.s + c * self.scale)[:, None])).float()

    def predict(self, X, W):
        return (X - self.mx) @ W + self.my


def fit(model, X_all, Y_new, q, N):
    dev = "cuda"
    tok = model.tokenizer
    V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
    strs = [tok.decode([i]) for i in range(V)]
    nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
    func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
    mask = nonword | func
    norm_s = [s.strip().lower() for s in strs]
    vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
    NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
    special_ids = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    special_ids |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
    junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
    junk_tok[[i for i in special_ids if i < V]] = True
    junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)

    def topk(v, k=10, bs=250, masked=True):
        out = []
        for i in range(0, v.shape[0], bs):
            lg = model.unembed(v[i:i + bs].to(dev).float()).float()
            if masked:
                lg = lg.masked_fill(mask, float("-inf"))
            out.append(lg.topk(k, -1).indices)
        return torch.cat(out)

    def agree(top, word):
        return (NORM[top] == NORM[word][:, None]).any(-1)

    Yk = [unit(Y_new[k].float()) for k in range(3)]
    raw1 = [topk(y, 1, masked=False)[:, 0] for y in Yk]
    junk = torch.stack([junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
    valid = cnt > 0
    word = topk(T, 10)[:, 0]
    c_ = Counter(norm_s[t] for t in word[NV + NT:].tolist())
    print(f"[{q}] junk vectors {int(junk.sum())}/{3 * N}; most common target words {c_.most_common(15)}", flush=True)
    tr_m = valid.clone(); tr_m[:NV + NT] = False
    va_m = valid.clone(); va_m[NV:] = False
    te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
    tr_i, va_i, te_i = (torch.nonzero(m)[:, 0] for m in (tr_m, va_m, te_m))
    maps, rows = {}, []
    for l in LAYERS:
        Xl = X_all[l].float().to(dev)
        r = Ridge(Xl[tr_i])
        r.setY(T[tr_i.cpu()].to(dev))
        sc = [agree(topk(r.predict(Xl[va_i], r.W(c))), word[va_i]).float().mean().item() for c in GRID]
        b = max(range(len(GRID)), key=lambda i: (sc[i], i))
        Wm = r.W(GRID[b])
        e10 = agree(topk(r.predict(Xl[te_i], Wm)), word[te_i]).float().mean().item()
        s = Wm.abs().max().item()
        maps[l] = dict(mx=r.mx.cpu(), W=(Wm / s).half().cpu(), scale=s, my=r.my.cpu(), c=GRID[b])
        rows.append(dict(layer=l, c=GRID[b], test_top10=e10))
        print(f"[{q}] L{l} c={GRID[b]} test top10 agreement {e10:.3f}", flush=True)
        del r, Xl
        torch.cuda.empty_cache()
    return maps, dict(rows=rows, junk_vectors=int(junk.sum()), common_targets=c_.most_common(30),
                      n_train=len(tr_i), n_val=len(va_i), n_test=len(te_i))


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    if NCHK:
        data = data[:NCHK]
    model = load_lens_model(None if WORD == "base" else WORD)
    X, Y = collect(model, data)
    if NCHK:  # reproduction check against the stored base collections
        Xs = torch.load(f"{R}/.hf27/ask2/acts_9b.pt", mmap=True)["X"]
        res = {}
        for l in LAYERS:
            res[f"X_L{l}_cos"] = torch.nn.functional.cosine_similarity(X[l].float(), Xs[l][:NCHK].float(), -1).mean().item()
        for k in range(3):
            ys = torch.load(f"{R}/.hf27/ask3/Y_9b_w{k}.pt", mmap=True)[:NCHK].float()
            res[f"Y_t3_w{k}_cos"] = torch.nn.functional.cosine_similarity(Y["t3"][k].float(), ys, -1).mean().item()
        a5 = f"{R}/.hf27/ask5/Y_9b_w0.pt"
        if os.path.exists(a5):
            ys = torch.load(a5, mmap=True)[:NCHK].float()
            res["Y_s5_w0_cos"] = torch.nn.functional.cosine_similarity(Y["s5"][0].float(), ys, -1).mean().item()
        print("REPRO", res, flush=True)
        jdump(res, f"{OUT}/alens_repro_check.json")
        return
    info = {}
    for q in QS:
        maps, info[q] = fit(model, X, Y[q], q, len(data))
        with open(f"{AD}/E_{WORD}_{q}.pt", "wb") as f:
            torch.save(maps, f)
    jdump(info, f"{OUT}/alens_fit_{WORD}.json")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
