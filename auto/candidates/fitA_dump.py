"""Viewer export (FIT split only): a row of the frozen A-lens family (auto/candidates/alens.py; env ALENS_BASE = none | JPP | JPP0 ...).
Dumps for every fit item and readout layer the filtered top-10 tokens and concept ranks, plus the unfiltered ones under "u",
and (w1, when the evaluator provides ctx.filters) the same under filter levels "f2" and "f3" (auto/filters.py; f3 top-10 =
best member of each of the top-10 token groups, ranks over groups, hit = the group holds a candidate id),
to runs/fitA_<tag>_<mode>.json. Run with the evaluator's default --filter f1 so "f" stays the benchmark filter;
the lens (and lambda) is the one alens.build picks under that level."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olens.readout import candidate_ranks, group_candidate_ranks, unembed_logits  # noqa: E402
import alens  # noqa: E402


def build(ctx):
    lens = alens.build(ctx); mode = os.environ.get("ALENS_BASE", "JPP")
    tok = ctx.model.tokenizer; f = ctx.fit; out = dict(layers=list(ctx.readout_layers), items={}, lam=getattr(lens, "lam", None))
    FL = getattr(ctx, "filters", None)
    levels = [("f", f.excluded_ids, None), ("u", None, None)]  # f = benchmark token filter, u = no filter
    if FL is not None:  # w1
        f2, f3 = FL.get("f2"), FL.get("f3")
        levels += [("f2", f2.excluded_ids, None), ("f3", f3.excluded_ids, f3.group_ids)]
        out["filters"] = dict(level_of_f=getattr(f, "filter_level", "f1"), n_masked_f2=len(f2.excluded_ids), n_groups_f3=f3.n_groups)
    for L in ctx.readout_layers:
        vec = lens.transport(f.resid[L], L)
        for key, excl, gid in levels:
            lg = unembed_logits(ctx.model, vec, excl)
            if gid is None:
                ranks = candidate_ranks(lg, f.candidates); top = lg.topk(10, dim=-1).indices.tolist()
            else:
                ranks = group_candidate_ranks(lg, f.candidates, gid); top = [FL.group_topk(lg[i], gid, 10) for i in range(lg.shape[0])]
            for i in range(len(f.item_names)):
                cand = {t for ids in f.candidates[i].values() for t in ids}
                d = out["items"].setdefault(f"{f.eval_slugs[i]}|{f.item_names[i]}", {})
                if gid is None:
                    hit = [int(t in cand) for t in top[i]]
                else:
                    cg = {int(gid[t]) for t in cand}; hit = [int(int(gid[t]) in cg) for t in top[i]]
                r = dict(top=[tok.decode([t]) for t in top[i]], hit=hit, rank=[ranks[i][k] for k in f.candidates[i]])
                if key == "f": d[str(L)] = r
                else: d[str(L)][key] = r
    tag = {2048: "2b", 4096: "9b", 5120: "27b"}[ctx.model.d_model]
    p = f"/workspace/o-lens-improvement/runs/fitA_{tag}_{mode}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False); print(f"[fitA] wrote {p}", flush=True)
    return lens
