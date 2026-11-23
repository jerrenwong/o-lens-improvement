"""Viewer export (FIT split only): a row of the frozen A-lens family (auto/candidates/alens.py; env ALENS_BASE = none | JPP).
Dumps for every fit item and readout layer the filtered top-10 tokens and the concept ranks to runs/fitA_<tag>_<mode>.json."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olens.readout import candidate_ranks, unembed_logits  # noqa: E402
import alens  # noqa: E402


def build(ctx):
    lens = alens.build(ctx); mode = os.environ.get("ALENS_BASE", "JPP")
    tok = ctx.model.tokenizer; f = ctx.fit; out = dict(layers=list(ctx.readout_layers), items={}, lam=getattr(lens, "lam", None))
    for L in ctx.readout_layers:
        lg = unembed_logits(ctx.model, lens.transport(f.resid[L], L), f.excluded_ids)
        ranks = candidate_ranks(lg, f.candidates); top = lg.topk(10, dim=-1).indices.tolist()
        for i in range(len(f.item_names)):
            cand = {t for ids in f.candidates[i].values() for t in ids}
            d = out["items"].setdefault(f"{f.eval_slugs[i]}|{f.item_names[i]}", {})
            d[str(L)] = dict(top=[tok.decode([t]) for t in top[i]], hit=[int(t in cand) for t in top[i]], rank=[ranks[i][k] for k in f.candidates[i]])
    tag = {2048: "2b", 4096: "9b", 5120: "27b"}[ctx.model.d_model]
    p = f"/workspace/o-lens-improvement/runs/fitA_{tag}_{mode}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False); print(f"[fitA] wrote {p}", flush=True)
    return lens
