"""lr1: score a saved lens file (env R2_LENS) as r2_lensfile.py does (h @ J[layer].T), and print its FIT-split
pass@10 (labelled ctx.fit items, same metric as the scorer: min rank over layers, per-item share, task macro)."""
import os

import torch
from workspace_lens.utils import load_lens_file


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def fit_score(ctx, lens, k):
    fit, model = ctx.fit, ctx.model
    V = model._lm_head.weight.shape[0]
    mask = torch.zeros(V, dtype=torch.bool, device=ctx.device)
    mask[fit.excluded_ids.to(ctx.device)] = True
    best = {}
    for l in ctx.readout_layers:
        lg = model.unembed(lens.transport(fit.resid[l].clone(), l)).float()
        lg = lg.masked_fill(mask[None, :lg.shape[1]], float("-inf"))
        for i, cands in enumerate(fit.candidates):
            for name, ids in cands.items():
                bl = lg[i, torch.as_tensor(ids, device=lg.device)].max()
                r = int((lg[i] > bl).sum()) + 1
                best[(i, name)] = min(best.get((i, name), 10**9), r)
    item = {}
    for (i, name), r in best.items():
        item.setdefault(i, []).append(float(r <= k))
    per = {}
    for i, h in item.items():
        per.setdefault(fit.eval_slugs[i], []).append(sum(h) / len(h))
    per = {e: sum(v) / len(v) for e, v in sorted(per.items())}
    return sum(per.values()) / len(per), per


def build(ctx):
    L = load_lens_file(os.environ["R2_LENS"], hf_model_name="Qwen/Qwen3.5-9B")
    lens = Lens({int(l): v.to(ctx.device).float() for l, v in L.jacobians_L_dict_FN.items()})
    for k in (10, 1):
        s, per = fit_score(ctx, lens, k)
        print(f"FIT_pass{k} {s:.4f} " + " ".join(f"{e} {v:.3f}" for e, v in per.items()), flush=True)
    return lens
