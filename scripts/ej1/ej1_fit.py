"""ej1: the elicited J-lens. J-lens estimator (averaged Jacobian, raw text, plain or J++ backward rules), but the
targets are the model's final-block states after an inline elicitation suffix appended to the passage.

Per passage P (tokens p_0..p_{n-1}), suffix S (k tokens), the model's own greedy continuation c_1..c_3 (teacher-forced):
  text     = P + S + c1 c2 c3                (no chat template, no BOS: as the J-lens fit)
  targets  T1: final-block state at S's last token;  T3: that one + the 3 continuation positions (summed)
  sources  S1: block-l output at P's last token;     S2: every passage position from the 4th token on
  E[l]     = sum over (passage, source pos) of the summed Jacobian d target / d source, / #(passage, source pos)
Control c1 (suffix "none"): S = the model's own greedy continuation of the same length k as " (i.e." -> same footing.
One run yields all four (S1|S2) x (T1|T3) matrices; even/odd passages are kept apart for split-half stability.
Prior: mean of unit(E h) over 500 block-l outputs at random positions of 500 other passages (saved, top tokens printed).

  python scripts/ej1/ej1_fit.py --model M --suffix ie|namely|thatis|iow|none --pool e1|wt --n 128 --lrp none|rlens --out DIR
"""
import argparse
import json
import os
import random
import sys
import time

if os.environ.get("EJ1_FLA") != "1":  # torch GatedDeltaNet path (as ja1; needed for the LRP surgery)
    sys.modules["fla"] = None
import torch as t  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.lrp import apply_lrp_rules_for_mode  # noqa: E402

R = "/workspace/o-lens-improvement"
LAYERS = {"Qwen/Qwen3.5-2B": [3, 6, 9, 12, 15, 18, 21], "Qwen/Qwen3.5-9B": [4, 8, 12, 16, 20, 24, 28]}
SUFFIXES = {"ie": " (i.e.", "namely": " (namely", "thatis": " — that is,", "iow": " (in other words,"}
SEED = 20261127
SNAPS = (128, 256)


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def passages(tok, pool, n, skip=0):
    """[(ids list)]: e1 = runs/e1/prefixes.json train rows (index >= 2000), shuffled; wt = WikiText windows
    (runs/r1_2b/prompts_raw.json) cut to <= 96 tokens at the last ' . ' sentence end (>= 24 tokens)."""
    if pool == "e1":
        d = json.load(open(f"{R}/runs/e1/prefixes.json"))[2000:]
        order = list(range(len(d)))
        random.Random(SEED).shuffle(order)
        return [(d[i]["ids"], d[i]["text"]) for i in order[skip:skip + n]]
    raw = json.load(open(f"{R}/runs/r1_2b/prompts_raw.json"))
    out = []
    for s in raw[1000:]:  # not the J-lens fit windows
        ids = tok(s, add_special_tokens=False).input_ids[:96]
        txt = tok.decode(ids)
        cut = txt.rfind(" . ")
        if cut < 0:
            continue
        ids = tok(txt[:cut + 2], add_special_tokens=False).input_ids
        if len(ids) >= 24:
            out.append((ids, txt[:cut + 2]))
    return out[skip:skip + n]


@t.no_grad()
def greedy(model, ids, k):
    ids = list(ids)
    for _ in range(k):
        h = model.forward(t.tensor([ids], device=model.input_device)).last_hidden_state[0, -1]
        ids.append(int(model._lm_head(h).argmax()))  # last_hidden_state is final-normed
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--suffix", required=True, choices=[*SUFFIXES, "none"])
    ap.add_argument("--pool", default="e1", choices=["e1", "wt"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--lrp", default="none")
    ap.add_argument("--rpp", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = get_hf_model(a.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    tok, d, B = model.tokenizer, model.d_model, a.rpp
    layers = LAYERS[a.model]
    tl = model.n_layers - 1
    kref = len(tok(SUFFIXES["ie"], add_special_tokens=False).input_ids)
    suf = None if a.suffix == "none" else tok(SUFFIXES[a.suffix], add_special_tokens=False).input_ids
    P = passages(tok, a.pool, a.n)
    assert len(P) == a.n, len(P)
    dev = model.input_device
    keys = [(s, tt, h) for s in ("S1", "S2") for tt in ("T1", "T3") for h in (0, 1)]
    sums = {k: {l: t.zeros(d, d, device=dev) for l in layers} for k in keys}
    cnt = {k: 0 for k in keys}
    recs = []
    bi = t.arange(B, device=dev)
    t0 = time.time()

    def save(n):
        tag = "" if n == a.n else f"_n{n}"
        stab = {}
        for s in ("S1", "S2"):
            for tt in ("T1", "T3"):
                E = {l: ((sums[(s, tt, 0)][l] + sums[(s, tt, 1)][l]) / (cnt[(s, tt, 0)] + cnt[(s, tt, 1)])) for l in layers}
                t.save({l: v.to(t.bfloat16).cpu() for l, v in E.items()}, f"{a.out}/E_{s}{tt}{tag}.pt")
                stab[f"{s}{tt}"] = {l: float(t.nn.functional.cosine_similarity(
                    sums[(s, tt, 0)][l].flatten(), sums[(s, tt, 1)][l].flatten(), dim=0)) for l in layers}
        dt = time.time() - t0
        info = dict(model=a.model, suffix=a.suffix, suffix_text=SUFFIXES.get(a.suffix), pool=a.pool, n=n, lrp=a.lrp,
                    seconds=dt, s_per_passage=dt / n, split_half_cos=stab, examples=recs[:12])
        json.dump(info, open(f"{a.out}/info{tag}.json", "w"), indent=1, ensure_ascii=False)
        print(f"[ej1] saved n={n} {dt:.0f}s ({dt / n:.1f}s/passage); split-half cos "
              + " ".join(f"{k}:{min(v.values()):.3f}-{max(v.values()):.3f}" for k, v in stab.items()), flush=True)

    for j, (pids, ptxt) in enumerate(P):
        n_p = len(pids)
        if suf is None:
            full = greedy(model, pids, kref + 3)
            k = kref
        else:
            full = greedy(model, pids + suf, 3)
            k = len(suf)
        T = len(full)
        tpos = {"T1": [n_p + k - 1], "T3": [n_p + k - 1, n_p + k, n_p + k + 1, n_p + k + 2]}
        assert tpos["T3"][-1] == T - 1
        srcs = {"S1": [n_p - 1], "S2": list(range(3, n_p))}
        h = j % 2
        recs.append(dict(passage=ptxt, elicited=tok.decode(full[n_p:]), n_tok=n_p))
        if j < 3:
            print(f"[ej1] example {j}: {tok.decode(full)!r} | targets after {tok.decode(full[:n_p + k])[-30:]!r}",
                  flush=True)
        ids = t.tensor([full], device=dev)
        with apply_lrp_rules_for_mode(model, a.lrp):
            with ActivationRecorder(model.layers, at=[*layers, tl], start_graph_at=min(layers)) as rec, t.enable_grad():
                model.forward(ids.expand(B, -1))
                target = rec.activations[tl]
                sources = [rec.activations[l] for l in layers]
            cot = t.zeros_like(target)
            starts = list(range(0, d, B))
            for ci, ds in enumerate(starts):
                nr = min(B, d - ds)
                for tt in ("T1", "T3"):
                    cot.zero_()
                    for q in tpos[tt]:
                        cot[bi[:nr], q, ds + bi[:nr]] = 1.0
                    last = ci == len(starts) - 1 and tt == "T3"
                    grads = t.autograd.grad(target, sources, grad_outputs=cot, retain_graph=not last)
                    for l, g in zip(layers, grads):
                        g = g[:nr].float()
                        for s in ("S1", "S2"):
                            sums[(s, tt, h)][l][ds:ds + nr] += g[:, srcs[s]].sum(1)
                    del grads
        for s in ("S1", "S2"):
            for tt in ("T1", "T3"):
                cnt[(s, tt, h)] += len(srcs[s])
        del target, sources, rec
        if j % 16 == 0:
            print(f"[ej1] {j + 1}/{a.n} {time.time() - t0:.0f}s, max alloc {t.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
        if j + 1 in SNAPS and j + 1 < a.n:
            save(j + 1)
    save(a.n)

    # suffix prior: mean unit(E h) over 500 unrelated activations (random positions of 500 other passages)
    Q = passages(tok, "e1", 500, skip=20000)
    rng = random.Random(1)
    H = {l: [] for l in layers}
    with t.no_grad():
        for ids, _ in Q:
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(t.tensor([ids], device=dev))
            p = rng.randrange(len(ids))
            for l in layers:
                H[l].append(rec.activations[l][0, p].float())
        H = {l: t.stack(v) for l, v in H.items()}
        prior, tops = {}, {}
        for s in ("S1", "S2"):
            for tt in ("T1", "T3"):
                E = t.load(f"{a.out}/E_{s}{tt}.pt")
                pr = {l: unit(H[l] @ E[l].to(dev).float().T).mean(0) for l in layers}
                prior[f"{s}{tt}"] = {l: v.cpu() for l, v in pr.items()}
                tops[f"{s}{tt}"] = {l: [tok.decode([i]) for i in model.unembed(v[None]).float()[0].topk(10).indices.tolist()]
                                    for l, v in pr.items()}
        t.save(prior, f"{a.out}/prior.pt")
        json.dump(tops, open(f"{a.out}/prior_top.json", "w"), indent=1, ensure_ascii=False)
        for k, v in tops.items():
            print(f"[ej1] prior {k} L{layers[3]}: {v[layers[3]]}", flush=True)
    print("[ej1] done", flush=True)


if __name__ == "__main__":
    main()
