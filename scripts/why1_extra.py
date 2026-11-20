"""why1 extra: one dev scoring of J++ + lam*unit(bag map c_all) with lam chosen on FIT by A's rule (0.2; ties ->
smaller), so the bag-of-words control is not handicapped by A's lam=0.5. Same standalone path as why1_analysis3."""
import json
import sys

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

from olens.readout import Bench, readout_rows  # noqa: E402
sys.path += [f"{R}/auto"]
from evaluate import pass_at_k  # noqa: E402

M, LAYERS, dev, LAM = "Qwen/Qwen3.5-9B", [4, 8, 12, 16, 20, 24, 28], "cuda", 0.2
torch.set_grad_enabled(False)


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


model = get_hf_model(M, attn_implementation="sdpa")
tok, hf = model.tokenizer, model._hf_model
EMB = hf.get_input_embeddings().weight
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
split = json.load(open(f"{R}/auto/split_9b.json"))
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
idx = sorted(pos[tuple(k)] for k in split["dev"])
devset = set(map(tuple, split["dev"]))
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json"))
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in devset}
mx, W, my = (t.to(dev) for t in torch.load("/root/why1/bagmaps.pt")["c_all"])
F = []
for i in idx:
    it = meta[(bench.eval_slugs[i], bench.item_names[i])]
    ids = tok("".join(rd["vocab"][x] for x in it["toks"]), add_special_tokens=False)["input_ids"]
    assert len(ids) == len(it["toks"])
    F.append(EMB[torch.tensor(ids[:it["pos"] + 1], device=EMB.device)].float().mean(0).to(dev))
mo = (torch.stack(F) - mx) @ W + my
si = torch.as_tensor(idx)
rows = {True: [], False: []}
for l in LAYERS:
    v = unit(bench.resid[l][si].to(dev).float() @ JPP[l].T) + LAM * unit(mo)
    for f in (True, False):
        rows[f] += readout_rows("why1_c_all_lam02", l, bench, v, model, filtered=f, items=idx, chunk_size=1)
p10, per = pass_at_k(rows[True], 10)
res = dict(pass10=p10, pass1=pass_at_k(rows[True], 1)[0], pass10_unf=pass_at_k(rows[False], 10)[0], per_task=per)
print("---", res, flush=True)
json.dump(res, open(f"{R}/runs/why1/q1extra_c_all_lam02.json", "w"), indent=1)
