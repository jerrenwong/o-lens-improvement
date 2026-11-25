"""w11 diagnostic (pod): cosine of per-token directions with the J-lens direction g_X = J^T (gamma * u_X), and between
variants, binned by N_X.   python scripts/w11/w11_cos.py <model> <tag> <vars csv>  -> runs/w11/cos_<tag>.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
from workspace_lens.utils import load_lens_file  # noqa: E402
sys.path += [str(R / "src"), str(R / "scripts/perf1")]
from lite_model import get_lite_model  # noqa: E402


def head_gamma(model):  # = auto/candidates/fm27_lib.head_gamma
    norm, head = model._final_norm, model._lm_head
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    p = next(norm.parameters())
    ones = torch.ones(1, head.weight.shape[1], dtype=torch.float32, device=p.device)
    return (norm(ones).float()[0] * (1.0 + eps) ** 0.5).to(head.weight.device)

model_name, tag, vars_ = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
J = {"2b": "artifacts/baselines/jlens_prefit.pt", "9b": "runs/v6_9b/jlens9b.pt"}[tag]
m = get_lite_model(model_name)
Jd = {int(l): v.cuda().float() for l, v in load_lens_file(str(R / J), hf_model_name=model_name).jacobians_L_dict_FN.items()}
WG = m._lm_head.weight.float() * head_gamma(m).float()[None]
meta = torch.load(R / f".hf27/w11/acc_{tag}_meta.pt")
N = meta["N"].cuda()
tabs = {v: torch.load(R / f".hf27/w11/acc_{tag}_{v}.pt") for v in vars_}
ids = meta["tok_ids"].cuda()
bins = [(1, 3), (3, 10), (10, 30), (30, 100), (100, 1000), (1000, 10 ** 9)]
out = {}
for l in tabs[vars_[0]]["layers"]:
    g = WG[ids] @ Jd[l]
    g = g / g.norm(dim=1, keepdim=True)
    D = {v: tabs[v]["dir"][l].cuda().float() for v in vars_}
    o = {}
    for lo, hi in bins:
        k = (N >= lo) & (N < hi)
        e = {f"cos({v},J)": float((D[v][k] * g[k]).sum(1).mean()) for v in vars_}
        if "c" in D:
            e.update({f"cos({v},c)": float((D[v][k] * D["c"][k]).sum(1).mean()) for v in vars_ if v != "c"})
        e["n"] = int(k.sum())
        o[f"{lo}-{hi}"] = e
    out[l] = o
    print(l, json.dumps(o))
(R / f"runs/w11/cos_{tag}.json").write_text(json.dumps(out, indent=1))
