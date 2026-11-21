"""Viewer export (FIT split only): J++ with the gated junk-direction removal (q8_gate). In build(), dump for every fit
item at the gated layers whether the gate fired and, where it did, the top-10 tokens and concept ranks (with and
without the token filter). Returns the q8 gate lens, so the dev score must equal q8_gate's."""
import json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_gate  # noqa: E402
from olens.readout import candidate_ranks, unembed_logits  # noqa: E402


def build(ctx):
    lens = q8_gate.build(ctx)
    tok = ctx.model.tokenizer
    f = ctx.fit
    out = dict(layers=sorted(lens.U), items={})
    for L in sorted(lens.U):
        h = f.resid[L]
        v0 = h @ ctx.JPP[L].T
        gate = torch.nn.functional.cosine_similarity(v0, lens.U[L].to(v0.dtype)[None], dim=-1) > q8_gate.THR
        v = lens.transport(h, L)
        for mode, ex in (("f", f.excluded_ids), ("u", None)):
            lg = unembed_logits(ctx.model, v, ex)
            ranks = candidate_ranks(lg, f.candidates)
            top = lg.topk(10, dim=-1).indices.tolist()
            for i in range(len(f.item_names)):
                if not bool(gate[i]):
                    continue
                cand = {t for ids in f.candidates[i].values() for t in ids}
                key = f"{f.eval_slugs[i]}|{f.item_names[i]}"
                d = out["items"].setdefault(key, {}).setdefault(str(L), {})
                d[mode] = dict(top=[tok.decode([t]) for t in top[i]], hit=[int(t in cand) for t in top[i]],
                               rank=[ranks[i][k] for k in f.candidates[i]])
    tag = {2048: "2b", 4096: "9b", 5120: "27b"}[ctx.model.d_model]
    p = f"/workspace/o-lens-improvement/runs/fitgate_{tag}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print(f"[fitgate] wrote {p}: gated items {len(out['items'])} of {len(f.item_names)} at layers {out['layers']}", flush=True)
    return lens
