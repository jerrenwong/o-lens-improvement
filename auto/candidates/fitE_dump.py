"""Viewer export (FIT split only): the ask-map read ALONE, v = unit(E h + b) (the map inside lens L; st1_lib "B0").
Dumps for every fit item and readout layer the filtered top-10 tokens and the concept ranks. Returns the map-alone lens."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olens.readout import candidate_ranks, unembed_logits  # noqa: E402
from st1_lib import build_alone  # noqa: E402


def build(ctx):
    lens = build_alone(ctx, "B0")
    tok = ctx.model.tokenizer; f = ctx.fit; out = dict(layers=list(ctx.readout_layers), items={})
    for L in ctx.readout_layers:
        lg = unembed_logits(ctx.model, lens.transport(f.resid[L], L), f.excluded_ids)
        ranks = candidate_ranks(lg, f.candidates); top = lg.topk(10, dim=-1).indices.tolist()
        for i in range(len(f.item_names)):
            cand = {t for ids in f.candidates[i].values() for t in ids}
            d = out["items"].setdefault(f"{f.eval_slugs[i]}|{f.item_names[i]}", {})
            d[str(L)] = dict(top=[tok.decode([t]) for t in top[i]], hit=[int(t in cand) for t in top[i]], rank=[ranks[i][k] for k in f.candidates[i]])
    tag = {2048: "2b", 4096: "9b", 5120: "27b"}[ctx.model.d_model]
    p = f"/workspace/o-lens-improvement/runs/fitE_{tag}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False); print(f"[fitE] wrote {p}", flush=True)
    return lens
