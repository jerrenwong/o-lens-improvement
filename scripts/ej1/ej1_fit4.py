"""ej1 fit4 (= fit3 + --targets T1,T3 | T3 | T1). ej1 fit3 (= fit2 + split-half history every 16 passages, E written every --every passages and kept at
64/128/192 in <suffix>/n<k>/). ej1 fit, several suffixes in ONE process (one model load; for 9B memory): same estimator as ej1_fit.py, each passage
is run once per suffix in --suffixes (e.g. ie,none). Writes <out>/<suffix>/E_<S><T>.pt, info.json, prior.pt,
prior_top.json after every SNAP passages (so a fit can be stopped early and still evaluated).
    python scripts/ej1/ej1_fit2.py --model Qwen/Qwen3.5-9B --suffixes ie,none --n 128 --rpp 32 --out DIR
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ej1_fit import LAYERS, SUFFIXES, greedy, passages, unit  # noqa: E402  (hides fla, as ej1_fit)
import torch as t  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.lrp import apply_lrp_rules_for_mode  # noqa: E402

EVERY = 16
KEEP = (64, 128, 192)  # E snapshots kept in <suffix>/n<k>/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--suffixes", required=True)
    ap.add_argument("--pool", default="e1")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--lrp", default="none")
    ap.add_argument("--rpp", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--targets", default="T1,T3")
    a = ap.parse_args()
    model = get_hf_model(a.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    tok, d, B = model.tokenizer, model.d_model, a.rpp
    layers = LAYERS[a.model]
    tl = model.n_layers - 1
    dev = model.input_device
    kref = len(tok(SUFFIXES["ie"], add_special_tokens=False).input_ids)
    sufs = a.suffixes.split(",")
    P = passages(tok, a.pool, a.n)
    # prior activations first (500 random positions of 500 other passages), kept on CPU
    rng = random.Random(1)
    H = {l: [] for l in layers}
    with t.no_grad():
        for ids, _ in passages(tok, "e1", 500, skip=20000):
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(t.tensor([ids], device=dev))
            p = rng.randrange(len(ids))
            for l in layers:
                H[l].append(rec.activations[l][0, p].float().cpu())
    H = {l: t.stack(v) for l, v in H.items()}
    TT = a.targets.split(",")
    keys = [(s, tt, h) for s in ("S1", "S2") for tt in TT for h in (0, 1)]
    st = {}
    for sf in sufs:
        os.makedirs(f"{a.out}/{sf}", exist_ok=True)
        st[sf] = dict(sums={k: {l: t.zeros(d, d, device=dev) for l in layers} for k in keys}, cnt={k: 0 for k in keys},
                      recs=[], secs=0.0)
    bi = t.arange(B, device=dev)

    def save(sf, n, write=True):
        S = st[sf]
        o = f"{a.out}/{sf}" + (f"/n{n}" if n in KEEP else "")
        os.makedirs(o, exist_ok=True)
        stab, prior, tops = {}, {}, {}
        for s in ("S1", "S2"):
            for tt in TT:
                E = {l: (S["sums"][(s, tt, 0)][l] + S["sums"][(s, tt, 1)][l]) / (S["cnt"][(s, tt, 0)] + S["cnt"][(s, tt, 1)])
                     for l in layers}
                stab[f"{s}{tt}"] = {l: float(t.nn.functional.cosine_similarity(
                    S["sums"][(s, tt, 0)][l].flatten(), S["sums"][(s, tt, 1)][l].flatten(), dim=0)) for l in layers}
                if not write:
                    continue
                t.save({l: v.to(t.bfloat16).cpu() for l, v in E.items()}, f"{o}/E_{s}{tt}.pt")
                with t.no_grad():
                    pr = {l: unit(H[l].to(dev) @ E[l].to(t.bfloat16).float().T).mean(0) for l in layers}
                    prior[f"{s}{tt}"] = {l: v.cpu() for l, v in pr.items()}
                    tops[f"{s}{tt}"] = {l: [tok.decode([i]) for i in model.unembed(v[None]).float()[0].topk(10).indices.tolist()]
                                        for l, v in pr.items()}
        S.setdefault("hist", {})[n] = stab
        json.dump(S["hist"], open(f"{a.out}/{sf}/stab_hist.json", "w"), indent=1)
        print(f"[ej1] {sf} n={n} split-half cos " + " ".join(f"{k}:" + ",".join(f"{x:.2f}" for x in v.values()) for k, v in stab.items()), flush=True)
        if not write:
            return
        t.save(prior, f"{o}/prior.pt")
        json.dump(tops, open(f"{o}/prior_top.json", "w"), indent=1, ensure_ascii=False)
        json.dump(dict(model=a.model, suffix=sf, suffix_text=SUFFIXES.get(sf), pool=a.pool, n=n, lrp=a.lrp, rpp=B,
                       seconds=S["secs"], s_per_passage=S["secs"] / n, split_half_cos=stab, examples=S["recs"][:12]),
                  open(f"{o}/info.json", "w"), indent=1, ensure_ascii=False)
        print(f"[ej1] {sf} saved n={n} ({S['secs'] / n:.1f}s/passage); split-half cos "
              + " ".join(f"{k}:{min(v.values()):.3f}-{max(v.values()):.3f}" for k, v in stab.items()), flush=True)

    for j, (pids, ptxt) in enumerate(P):
        n_p = len(pids)
        for sf in sufs:
            S = st[sf]
            t0 = time.time()
            if sf == "none":
                full, k = greedy(model, pids, kref + 3), kref
            else:
                sids = tok(SUFFIXES[sf], add_special_tokens=False).input_ids
                full, k = greedy(model, pids + sids, 3), len(sids)
            tpos = {"T1": [n_p + k - 1], "T3": [n_p + k - 1, n_p + k, n_p + k + 1, n_p + k + 2]}
            srcs = {"S1": [n_p - 1], "S2": list(range(3, n_p))}
            h = j % 2
            S["recs"].append(dict(passage=ptxt, elicited=tok.decode(full[n_p:]), n_tok=n_p))
            if j < 2:
                print(f"[ej1] {sf} example {j}: {tok.decode(full)!r}", flush=True)
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
                    for tt in TT:
                        cot.zero_()
                        for q in tpos[tt]:
                            cot[bi[:nr], q, ds + bi[:nr]] = 1.0
                        last = ci == len(starts) - 1 and tt == TT[-1]
                        grads = t.autograd.grad(target, sources, grad_outputs=cot, retain_graph=not last)
                        for l, g in zip(layers, grads):
                            g = g[:nr].float()
                            for s in ("S1", "S2"):
                                S["sums"][(s, tt, h)][l][ds:ds + nr] += g[:, srcs[s]].sum(1)
                        del grads
            for s in ("S1", "S2"):
                for tt in TT:
                    S["cnt"][(s, tt, h)] += len(srcs[s])
            del target, sources, rec, cot
            t.cuda.synchronize()
            S["secs"] += time.time() - t0
        if j % 4 == 0:
            print(f"[ej1] {j + 1}/{a.n} " + " ".join(f"{sf} {st[sf]['secs'] / (j + 1):.0f}s/p" for sf in sufs)
                  + f", max alloc {t.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if (j + 1) % EVERY == 0 or j + 1 == a.n:
            for sf in sufs:
                save(sf, j + 1, write=(j + 1) % a.every == 0 or j + 1 == a.n)
    print("[ej1] done", flush=True)


if __name__ == "__main__":
    main()
