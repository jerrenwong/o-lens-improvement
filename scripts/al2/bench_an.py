"""al2 benchmark analysis, 9B DEV split only (fit+dev item text from the viewer export, as auto/qual3.py / why1).
Uses the scorer's own functions (olens.readout.readout_rows, evaluate.pass_at_k).
 (1) per-task dev pass@10 / pass@1 of M_all ALONE (and J++, mix for reference)
 (2) association: per-layer hit counts (rank <= 10, filtered) for J++, M_all, mix
 (3) top-8 tokens of J++ and M_all at L20, L28 for 10 dev association prompts, with the concept's rank.
argv[1] = lam (the one al2_T chose on FIT)."""
import json, sys
from collections import Counter
import torch
R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402
from olens.readout import Bench, readout_rows  # noqa: E402
sys.path += [f"{R}/auto", f"{R}/auto/candidates"]
from evaluate import pass_at_k  # noqa: E402
from e2_lib import ELens, load_maps  # noqa: E402

LAM = float(sys.argv[1])
M = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
dev = "cuda"
torch.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
MAPS = load_maps(f"{R}/.hf27/al2/M_all.pt", dev)
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
split = json.load(open(f"{R}/auto/split_9b.json"))
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
dev_idx = sorted(pos[tuple(k)] for k in split["dev"])
si = torch.as_tensor(dev_idx, dtype=torch.long)
rr = {l: v[si].to(dev).float() for l, v in bench.resid.items()}
lenses = {"J++": ELens(JPP, MAPS, "mix", 0.0), "M_all": ELens(JPP, MAPS, "alone"), "mix": ELens(JPP, MAPS, "mix", LAM)}
vecs = {n: {l: L.transport(rr[l].clone(), l) for l in LAYERS} for n, L in lenses.items()}
out = {"lam": LAM, "n_dev": len(dev_idx)}
for n, vs in vecs.items():
    rows = []
    for l, v in vs.items():
        rows += readout_rows(n, l, bench, v, model, filtered=True, items=dev_idx, chunk_size=1)
    p10, per = pass_at_k(rows, 10)
    p1, per1 = pass_at_k(rows, 1)
    out[n] = dict(pass10=p10, pass1=p1, per_task10=per, per_task1=per1)
    a = [r for r in rows if r["eval"] == "association"]
    out[n]["assoc_layer_hits10"] = {l: sum(r["rank"] <= 10 for r in a if r["layer"] == l) for l in LAYERS}
    best = {}
    for r in a:
        k = (r["item"], r["intermediate"]); best[k] = min(best.get(k, 10**9), r["rank"])
    out[n]["assoc_any_hits10"] = sum(v <= 10 for v in best.values())
    out[n]["assoc_n"] = len(best)
    print(n, json.dumps(out[n]), flush=True)
# (3) qualitative, 10 dev association prompts
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json"))
RV = rd["vocab"]
allowed = set(map(tuple, split["fit"])) | set(map(tuple, split["dev"]))
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in allowed}
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
exc = torch.zeros(V, dtype=torch.bool, device=dev)
exc[bench.excluded_ids.to(dev)] = True
assoc = [i for i in dev_idx if bench.eval_slugs[i] == "association"][:10]
qual = []
for i in assoc:
    k = (bench.eval_slugs[i], bench.item_names[i])
    it = meta[k]
    text = "".join(RV[x] for x in it["toks"][max(0, it["pos"] - 40):it["pos"] + 1])
    rec = dict(item=k[1], text=text, concepts={})
    for concept, cid in bench.candidates[i].items():
        cid = torch.as_tensor(cid, device=dev).view(-1)
        rec["concepts"][concept] = {}
        for l in (20, 28):
            h = bench.resid[l][i:i + 1].to(dev).float()
            for n in ("J++", "M_all"):
                lg = model.unembed(lenses[n].transport(h.clone(), l)).float()[0, :V].masked_fill(exc, float("-inf"))
                rank = int((lg > lg[cid].max()).sum()) + 1
                rec["concepts"][concept][f"{n}@L{l}"] = dict(rank=rank, top=[strs[j].strip() for j in lg.topk(8).indices.tolist()])
    qual.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
out["qual"] = qual
json.dump(out, open(f"{R}/.hf27/al2/bench_an.json", "w"), indent=1, ensure_ascii=False)
print("BENCH_AN_DONE")
