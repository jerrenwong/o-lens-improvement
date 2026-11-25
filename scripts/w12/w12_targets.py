"""w12 targets from the collected top-50 final distributions (scripts/w12/w12_collect.py meta.pt).
Per position: junk tokens removed (f4 token rule as scripts/w9/w9_targets._allowed: no letter/digit, special/added
tokens, tokens containing @ & < >), renormalised over the kept top-50 tokens. Per sample at position t:
  T1   : the distribution at t (the model's next-token prediction)
  T4   : mean of the distributions at t .. t+3
  T16  : mean over t .. t+15
  T16d : weights 0.7^j, j = 0..15 (normalised)
then top-10, renormalised. Output: DIR/targets.pt {name: (ids [N,10] int32, p [N,10] fp16)}, DIR/targets_stats.json
Usage: python w12_targets.py <model> DIR"""
import importlib.util
import json
import sys

import torch

R = "/workspace/o-lens-improvement"
M, D = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402

dev = "cuda"
model = get_lite_model(M)
V = model._lm_head.weight.shape[0]
fs = importlib.util.spec_from_file_location("w12_filters", f"{R}/auto/filters.py")
FM = importlib.util.module_from_spec(fs); fs.loader.exec_module(FM)
strs = FM.decode_all(model.tokenizer, V)
SPEC = FM.special_ids(model.tokenizer)
bad = ("@",) + tuple(FM.MARKUP_CHARS)
allowed = torch.tensor([any(c.isalnum() for c in s) and i not in SPEC and not any(c in s for c in bad)
                        for i, s in enumerate(strs)], device=dev)
meta = torch.load(f"{D}/meta.pt")
Fid, Fp, gpos = meta["Fid"], meta["Fp"], meta["gpos"]
N = len(gpos)
W = {"T1": [1.0], "T4": [0.25] * 4, "T16": [1 / 16] * 16}
w = torch.tensor([0.7 ** j for j in range(16)]); W["T16d"] = (w / w.sum()).tolist()
out = {k: ([], []) for k in W}
stats = {k: dict(mass_kept=0.0, top1_prob=0.0) for k in W}
junkmass = 0.0
CH = 2048
for c0 in range(0, N, CH):
    g = gpos[c0:c0 + CH]
    n = len(g)
    ix = g[:, None] + torch.arange(16)[None]  # [n,16]
    ii = Fid[ix.reshape(-1)].to(dev).long().view(n, 16, 50)
    pp = Fp[ix.reshape(-1)].to(dev).float().view(n, 16, 50)
    keep = allowed[ii]
    junkmass += (pp * ~keep).sum(-1)[:, 0].sum().item()
    pp = pp * keep
    pp = pp / pp.sum(-1, keepdim=True).clamp_min(1e-12)
    for k, wk in W.items():
        h = len(wk)
        Dn = torch.zeros(n, V, device=dev)
        wt = torch.tensor(wk, device=dev)[None, :, None]
        Dn.scatter_add_(1, ii[:, :h].reshape(n, -1), (pp[:, :h] * wt).reshape(n, -1))
        vv, vi = Dn.topk(10, -1)
        stats[k]["mass_kept"] += vv.sum().item()
        vv = vv / vv.sum(-1, keepdim=True).clamp_min(1e-12)  # all-junk rows stay all-zero (dropped in training)
        stats[k].setdefault("n_empty", 0); stats[k]["n_empty"] += int((vv.sum(-1) == 0).sum())
        stats[k]["top1_prob"] += vv[:, 0].sum().item()
        out[k][0].append(vi.int().cpu()); out[k][1].append(vv.half().cpu())
res = {k: (torch.cat(a), torch.cat(b)) for k, (a, b) in out.items()}
torch.save(res, f"{D}/targets.pt")
st = {k: {kk: vv / N for kk, vv in v.items()} for k, v in stats.items()}
st["junk_mass_top50_at_t"] = junkmass / N
st["N"] = N
json.dump(st, open(f"{D}/targets_stats.json", "w"), indent=1)
print(json.dumps(st), flush=True)
for k in W:
    print(k, [strs[i] for i in res[k][0][0].tolist()], res[k][1][0].tolist())
