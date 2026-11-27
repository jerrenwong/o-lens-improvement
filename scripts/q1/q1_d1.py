"""q1 D1: ask ceilings. For each item text (bench fit + dev, probes), ask the model directly and record the f4 word-group
rank of the hidden concept in the model's own answer distribution (next token at the answer position):
  q1_topic w0..w2 (+ "mean": unembed of the mean of the unit final states = the A-lens topic target),
  q2_unstated w0..w2 (+ mean), q3..q7 new quoted questions (one wording each),
  s1..s4 inline suffixes on the raw passage (no template): next token, and the 3 greedy steps (min rank, union top 10).
    python scripts/q1/q1_d1.py [model]      -> runs/q1/d1_<tag>.json
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/q1")
import q1_lib2 as Q  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = Q.TAGS[MODEL]


def main():
    t0 = time.time()
    model, cfg, bench, FL, F = Q.setup(MODEL)
    tok = model.tokenizer
    rows, qd = Q.load_items(model, bench, TAG)
    spare = {t for r in rows if r["set"] == "probes" for t in r["ids"]}
    S = Q.Scorer(F, spare)
    dev = "cuda"
    # few-shot hygiene: gold not in its passage; gold not a probe / bench concept
    concepts = {(r["concept"] or "").lower() for r in rows} | {a.lower() for r in rows for a in r.get("accepted", [])}
    for k, (_, _, lab, ex) in Q.NEW_Q.items():
        for t, w in ex:
            assert w.lower() not in t.lower(), (k, w)
            if w.lower() in concepts:
                print(f"WARNING few-shot gold {w!r} ({k}) is also an item concept", flush=True)
    specs = {}
    for k, sp in enumerate(Q.topic_specs()):
        specs[f"q1_topic_w{k}"] = sp
    for k, sp in enumerate(Q.unstated_specs()):
        specs[f"q2_unstated_w{k}"] = sp
    specs.update(Q.NEW_Q)
    texts = list(dict.fromkeys(r["text"] for r in rows))
    print(f"{len(rows)} rows, {len(texts)} texts, {len(specs)} quoted prompts + {len(Q.SUFFIXES)} suffixes per text",
          flush=True)
    for k, sp in specs.items():
        print(f"--- {k} prompt example:\n{Q.quoted_prompt(tok, sp, texts[0].strip(chr(10)))!r}", flush=True)
    out = {}  # text -> cond -> {lg masked (gpu, temp), top}
    res_text = {}
    for ti, text in enumerate(texts):
        conds = {}
        fin = {}
        for k, sp in specs.items():
            s = Q.quoted_prompt(tok, sp, text.strip("\n"))
            ids = torch.tensor([tok(s, add_special_tokens=False).input_ids], device=dev)
            y = Q.final_state(model, ids)[0]
            fin[k] = y
            conds[k] = S.masked(model.unembed(y[None]).float()[0])
        for q in ("q1_topic", "q2_unstated"):
            ys = [fin[f"{q}_w{k}"] for k in range(3)]
            T = sum(y / y.pow(2).mean().sqrt() for y in ys) / 3
            conds[f"{q}_mean"] = S.masked(model.unembed(T[None]).float()[0])
        gens = {}
        for k, suf in Q.SUFFIXES.items():
            ids = model.encode(text + suf)
            g = []
            for step in range(3):
                y = Q.final_state(model, ids)
                lg = model.unembed(y).float()[0]
                conds[f"{k}_g{step}"] = S.masked(lg)
                nt = int(lg[: S.mask.shape[0]].argmax())
                g.append(nt)
                ids = torch.cat([ids, torch.tensor([[nt]], device=ids.device)], 1)
            gens[k] = tok.decode(g)
        out[text] = conds
        res_text[text] = dict(top={k: [tok.decode([t]) for t in S.top(v)] for k, v in conds.items()}, gen=gens)
        # rows of this text
        for r in rows:
            if r["text"] != text:
                continue
            rk = {k: S.rank(v, r["ids"]) for k, v in conds.items()}
            for k in Q.SUFFIXES:
                rs = [rk[f"{k}_g{s}"] for s in range(3)]
                rk[f"{k}_g3"] = None if rs[0] is None else min(rs)
            r["ranks"] = rk
        del out[text]
        if ti % 25 == 0:
            print(f"{ti}/{len(texts)} texts {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        r.pop("bi", None)
    json.dump(dict(model=MODEL, conds=list(specs) + ["q1_topic_mean", "q2_unstated_mean"], suffixes=Q.SUFFIXES,
                   new_questions={k: v[2] for k, v in Q.NEW_Q.items()}, rows=rows, texts=res_text),
              open(f"{Q.R}/runs/q1/d1_{TAG}.json", "w"), ensure_ascii=False)
    print(f"wrote runs/q1/d1_{TAG}.json {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
