"""W4 step 3: read the organism (and the base model on the same tokens) with lenses.
    python scripts/w4/read.py <word> [tag]   (tag names adapter taboo_<tag>, maps E_<tag>_*, games_<tag>.json; default = word)
Sequences (from runs/w4/games_<word>.json, written by checks.py):
  P1 = 40 openers + 20 direct demands, prompt = chat template + generation prompt (thinking off, empty think block);
       positions = last token of the user content, then every token the template adds before the first reply token.
  P2 = first 30 games, hint j teacher-forced exactly as generated: gen_prompt(game[:2j]) + hint_j token ids;
       positions = every hint token.
States: base model (no LoRA) and organism (LoRA merged in memory), same token ids.
Lenses (all decoded with model.unembed = final norm + LM head; LoRA does not touch them: checked and recorded):
  logit (h itself), J, JPP, A (base maps), JPP+A (lam 0.5), and if present A_org / JPP+A_org (organism-refit maps
  .hf27/w4/E_<word>_{t3,s5}.pt), Jorg = J-lens refitted on the organism (.hf27/w4/jlens_<word>.pt).
Ranks of 23 targets (3 secret words + 20 random nouns, single-token forms) under
  F1 = benchmark mask (bench.excluded_ids: tokens with no letter or digit), rank over tokens, best form;
  F3 = F1 + special/added tokens + tokens with "@", then tokens grouped (case, leading space/underscore, plural s),
       rank over groups.
Also: the answer distribution of each state to the six A-lens prompts (t3, s5 x 3 wordings) on the 90 hint passages:
probability mass on each secret word's forms and top-1 token.
Out: .hf27/w4/reads_<word>.pt (full rank tensors, P1 top-10 ids), runs/w4/ranks_<word>.json (per-prompt ranks)."""
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, BASE, LAYERS, NOUNS20, OUT, R, WORDS, gen_prompt, jdump, word_forms  # noqa: E402
from collect_fit import QS, build  # noqa: E402

WORD = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else WORD
LAM = 0.5
dev = "cuda"
TARGETS = WORDS + NOUNS20


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def load_maps(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items()}


def emap(M, h, l):
    mx, W, my = M[l]
    return (h - mx) @ W + my


def main():
    import jlens
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from olens.readout import Bench
    from workspace_lens.utils import load_lens_file
    torch.set_grad_enabled(False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    G = json.load(open(f"{OUT}/games_{TAG}.json"))

    # ---- sequences and positions
    seqs = []  # (set, unit, ids, positions)
    for u, p in enumerate(G["openers"] + G["demands"]):
        s = gen_prompt(tok, [dict(role="user", content=p)])
        ids = tok(s, add_special_tokens=False).input_ids
        head = tok(f"<|im_start|>user\n{p}", add_special_tokens=False).input_ids
        assert ids[:len(head)] == head, p
        seqs.append(("P1", u, ids, list(range(len(head) - 1, len(ids)))))
    hints = []
    for g_i, g in enumerate(G["games"][:30]):
        for j in range(3):
            pr = tok(gen_prompt(tok, g[:2 * j + 1]), add_special_tokens=False).input_ids
            hi = tok(g[2 * j + 1]["content"], add_special_tokens=False).input_ids
            seqs.append(("P2", g_i * 3 + j, pr + hi, list(range(len(pr), len(pr) + len(hi)))))
            hints.append(g[2 * j + 1]["content"])
    pos_meta = [(s, u, i, tok.decode([ids[i]])) for s, u, ids, ps in seqs for i in ps]
    sets = torch.tensor([0 if m[0] == "P1" else 1 for m in pos_meta])
    units = torch.tensor([m[1] for m in pos_meta])
    print(f"positions: P1 {int((sets == 0).sum())} P2 {int((sets == 1).sum())}; P1 example tokens "
          f"{[m[3] for m in pos_meta if m[0] == 'P1' and m[1] == 0]}", flush=True)

    # ---- vocab, masks, groups, targets
    V = hf.lm_head.weight.shape[0]
    strs = [tok.decode([i]) for i in range(V)]
    bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
    F1 = torch.zeros(V, dtype=torch.bool)
    F1[bench.excluded_ids[bench.excluded_ids < V]] = True
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    F3 = F1.clone()
    F3[[i for i in special if i < V]] = True
    F3 |= torch.tensor(["@" in s for s in strs])

    def norm(s):
        s = s.strip().lstrip("_▁").lower()
        return s[:-1] if len(s) > 3 and s.endswith("s") else s
    gkey = [norm(s) for s in strs]
    gidx = {k: i for i, k in enumerate(dict.fromkeys(gkey))}
    GID = torch.tensor([gidx[k] for k in gkey], device=dev)
    NG = len(gidx)
    forms = {w: word_forms(tok, w) for w in TARGETS}
    tg = torch.tensor([gidx[norm(w)] for w in TARGETS], device=dev)
    for w in TARGETS:
        assert forms[w], w
        assert all(not F3[i] for i in forms[w]), w
    print("forms", {w: [(i, strs[i]) for i in forms[w]] for w in WORDS}, flush=True)
    F1d, F3d = F1.to(dev), F3.to(dev)
    maxf = max(len(v) for v in forms.values())
    FT = torch.tensor([forms[w] + [forms[w][0]] * (maxf - len(forms[w])) for w in TARGETS], device=dev)  # [T, maxf]

    # ---- lenses
    def lens_dict(path):
        L = load_lens_file(path, hf_model_name=BASE)
        return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items() if int(l) in LAYERS}
    J = lens_dict(f"{R}/runs/v6_9b/jlens9b.pt")
    JPP = lens_dict(f"{R}/artifacts/baselines/jpp9b_prefit.pt")
    Eb = (load_maps(f"{R}/.hf27/ask3/E_9b_t3.pt"), load_maps(f"{R}/.hf27/ask5/E_9b_s5.pt"))
    Eo = None
    if os.path.exists(f"{AD}/E_{TAG}_t3.pt") and os.path.exists(f"{AD}/E_{TAG}_s5.pt"):
        Eo = (load_maps(f"{AD}/E_{TAG}_t3.pt"), load_maps(f"{AD}/E_{TAG}_s5.pt"))
    Jo = lens_dict(f"{AD}/jlens_{TAG}.pt") if os.path.exists(f"{AD}/jlens_{TAG}.pt") else None

    def A(E, h, l):
        return unit(unit(emap(E[0], h, l)) + unit(emap(E[1], h, l)))
    lenses = {"logit": lambda h, l: h, "J": lambda h, l: h @ J[l].T, "JPP": lambda h, l: h @ JPP[l].T,
              "A": lambda h, l: A(Eb, h, l), "JPP+A": lambda h, l: unit(h @ JPP[l].T) + LAM * A(Eb, h, l)}
    if Eo:
        lenses["A_org"] = lambda h, l: A(Eo, h, l)
        lenses["JPP+A_org"] = lambda h, l: unit(h @ JPP[l].T) + LAM * A(Eo, h, l)
    if Jo:
        lenses["Jorg"] = lambda h, l: h @ Jo[l].T
    print("lenses", list(lenses), f"{time.time() - t0:.0f}s", flush=True)

    # ---- activations and A-question answers per state
    def run_state(model):
        lm = jlens.from_hf(model, tok)
        H = {l: [] for l in LAYERS}
        for _, _, ids, ps in seqs:
            rec = {}
            hk = [lm.layers[l].register_forward_hook(
                (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[0]))(l))
                for l in LAYERS]
            pad = -(-len(ids) // 64) * 64 - len(ids)  # right pad to a 64 bucket (causal: earlier positions unchanged)
            try:
                lm.forward(torch.tensor([ids + [tok.pad_token_id or 0] * pad], device=dev))
            finally:
                for h in hk:
                    h.remove()
            for l in LAYERS:
                H[l].append(rec[l][ps].float())
        H = {l: torch.cat(v) for l, v in H.items()}
        ans = {}
        for q, W in QS.items():
            for k in range(3):
                mass, top1 = [], []
                for t in hints:
                    ids = tok(build(tok, W, k, t), add_special_tokens=False).input_ids
                    lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
                    pr = lg.softmax(-1)
                    mass.append([pr[forms[w]].sum().item() for w in WORDS])
                    top1.append(strs[int(lg.argmax())])
                ans[f"{q}_w{k}"] = dict(mass=mass, top1=top1)
        return lm, H, ans

    nP1 = int((sets == 0).sum())

    def decode_ranks(lm, H):
        out = {}
        for name, fn in lenses.items():
            r1 = torch.zeros(len(LAYERS), len(pos_meta), len(TARGETS), dtype=torch.int32)
            r3 = torch.zeros_like(r1)
            top = torch.zeros(len(LAYERS), int((sets == 0).sum()), 10, dtype=torch.int32)
            for li, l in enumerate(LAYERS):
                for b in range(0, len(pos_meta), 128):
                    h = H[l][b:b + 128]
                    lg = lm.unembed(fn(h, l)).float()
                    l1 = lg.masked_fill(F1d, float("-inf"))
                    tv = l1[:, FT].amax(-1)  # [n, T] best form
                    r1[li, b:b + len(h)] = torch.stack([(l1 > tv[:, t:t + 1]).sum(-1) + 1
                                                        for t in range(len(TARGETS))], -1).int().cpu()
                    l3 = lg.masked_fill(F3d, float("-inf"))
                    gm = torch.full((len(h), NG), float("-inf"), device=dev).scatter_reduce_(
                        1, GID.expand(len(h), -1), l3, "amax")
                    gv = gm[:, tg]
                    r3[li, b:b + len(h)] = torch.stack([(gm > gv[:, t:t + 1]).sum(-1) + 1
                                                        for t in range(len(TARGETS))], -1).int().cpu()
                    if b < nP1:  # P1 positions come first in pos_meta
                        e = min(len(h), nP1 - b)
                        top[li, b:b + e] = l1[:e].topk(10, -1).indices.int().cpu()
            out[name] = dict(F1=r1, F3=r3, top10_P1=top)
        return out

    res = {}
    lm, H, ans_b = run_state(hf)
    res["base"] = decode_ranks(lm, H)
    print(f"base done {time.time() - t0:.0f}s", flush=True)
    del H, lm
    keep = {n: p.detach().clone() for n, p in hf.named_parameters() if "norm" in n and "layers" not in n}
    keep["lm_head"] = hf.lm_head.weight.detach().clone()
    org = PeftModel.from_pretrained(hf, f"{AD}/taboo_{TAG}").merge_and_unload().eval()
    now = dict(org.named_parameters())
    unchanged = {n: bool(torch.equal(v, (org.lm_head.weight if n == "lm_head" else now[n]))) for n, v in keep.items()}
    print("final norm / LM head unchanged by the LoRA:", unchanged, flush=True)
    lm, H, ans_o = run_state(org)
    res["org"] = decode_ranks(lm, H)
    print(f"org done {time.time() - t0:.0f}s", flush=True)

    meta = dict(word=WORD, tag=TAG, layers=LAYERS, targets=TARGETS, lenses=list(lenses), lam=LAM,
                forms={w: [[i, strs[i]] for i in forms[w]] for w in TARGETS}, unchanged_norm_head=unchanged,
                pos_meta=pos_meta, n_units={"P1": len(G["openers"]) + len(G["demands"]), "P2": 90},
                n_openers=len(G["openers"]), hints=hints, answers=dict(base=ans_b, org=ans_o))
    with open(f"{AD}/reads_{TAG}.pt", "wb") as f:
        torch.save(dict(meta=meta, sets=sets, units=units, res=res), f)
    # compact JSON: per unit, per layer, best rank over the unit's positions, 3 secret words; nouns best overall
    js = dict(meta={k: v for k, v in meta.items() if k != "pos_meta"}, ranks={})
    for st in res:
        for ln in res[st]:
            for F in ("F1", "F3"):
                r = res[st][ln][F]
                for si, sn in enumerate(("P1", "P2")):
                    m = sets == si
                    us = units[m]
                    rr = r[:, m]
                    per = []
                    for u in us.unique().tolist():
                        x = rr[:, us == u]  # [L, pos, T]
                        per.append(dict(unit=u, words_by_layer=x[:, :, :3].amin(1).tolist(),
                                        nouns_best=x[:, :, 3:].amin(0).amin(0).tolist()))
                    js["ranks"][f"{st}|{ln}|{F}|{sn}"] = per
    with open(f"{OUT}/ranks_{TAG}.json", "w") as f:
        json.dump(js, f)
    print("saved", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
