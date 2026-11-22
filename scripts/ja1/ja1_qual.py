"""ja1 qualitative check on the FIT split only: 8 fit rows (2 per task, first 4 tasks) whose hidden concept J++ misses
(best filtered rank over layers > 10). Prints filtered top-8 tokens at two mid layers for J++, the ridge ask-map
alone (unit(E h + b) of fm27_lib.Base / m2b Base2B) and a J_ask lens alone (dict file).

    python scripts/ja1/ja1_qual.py --model Qwen/Qwen3.5-2B --J runs/ja1/2b/lens/X.pt --layers 9 12
"""
import argparse
import json
import sys
from types import SimpleNamespace

import torch

R = "/workspace/o-lens-improvement"
sys.path[:0] = [f"{R}/src", f"{R}/scripts/perf1"]
import workspace_lens  # noqa: E402,F401  (before auto/candidates, which holds a module named jlens.py)
from olens.readout import Bench, candidate_ranks  # noqa: E402
sys.path.append(f"{R}/auto/candidates")
from fm27_lib import NEG, best_of, logits  # noqa: E402
from lite_model import get_lite_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

MODELS = {"Qwen/Qwen3.5-9B": ("9b", "artifacts/bench/qwen3.5-9b.pt", "artifacts/baselines/jpp9b_prefit.pt"),
          "Qwen/Qwen3.5-2B": ("2b", "artifacts/bench/qwen3.5-2b.pt", "artifacts/baselines/jpp_prefit.pt")}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--J", required=True)
ap.add_argument("--layers", type=int, nargs=2, required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_grad_enabled(False)
tag, bench_p, jpp_p = MODELS[a.model]
dev = "cuda"
bench = Bench.load(f"{R}/{bench_p}")
fit_idx = bench.index("fit")
model = get_lite_model(a.model)
JPP = {int(l): v.to(dev).float() for l, v in load_lens_file(f"{R}/{jpp_p}", hf_model_name=a.model).jacobians_L_dict_FN.items()}
JA = {int(l): v.to(dev).float() for l, v in torch.load(a.J, weights_only=True).items()}
fi = torch.as_tensor(fit_idx)
resid = {l: v[fi].to(dev).float() for l, v in bench.resid.items()}
cands = [bench.candidates[i] for i in fit_idx]
slugs = [bench.eval_slugs[i] for i in fit_idx]
names = [bench.item_names[i] for i in fit_idx]
ctx = SimpleNamespace(JPP=JPP, device=dev, model=model)
if tag == "2b":
    from m2b_lib import Base2B
    ask = Base2B(ctx)
else:
    from fm27_lib import Base
    ask = Base(ctx)
V = model._lm_head.weight.shape[0]
mask = torch.zeros(V, dtype=torch.bool, device=model._lm_head.weight.device)
mask[bench.excluded_ids.to(mask.device)] = True
tok = model.tokenizer

rk = {}
for l in bench.layers:
    rk[l] = candidate_ranks(logits(model, resid[l] @ JPP[l].T).masked_fill(mask, NEG), cands)
best = best_of(rk)
picked, per = [], {}
for p, d in enumerate(best):
    miss = [k for k, r in d.items() if r > 10]
    if miss and per.get(slugs[p], 0) < 2:
        per[slugs[p]] = per.get(slugs[p], 0) + 1
        picked.append((p, miss))
tasks = sorted(per)[:4]
picked = [x for x in picked if slugs[x[0]] in tasks]

lines = []
for p, miss in picked:
    lines.append(f"\n## {slugs[p]} / {names[p]}  hidden (missed by J++): "
                 + ", ".join(f"{k} [J++ best rank {best[p][k]}]" for k in miss))
    for l in a.layers:
        h = resid[l][p:p + 1]
        uj, ue = ask.parts(h, l)
        for lab, v in (("J++", uj), ("ask-map", ue), ("J_ask", h @ JA[l].T)):
            L = logits(model, v).masked_fill(mask, NEG)
            top = L[0].topk(8).indices.tolist()
            r = candidate_ranks(L, [cands[p]])[0]
            lines.append(f"  L{l:2d} {lab:8s} {' | '.join(repr(tok.decode([i]).strip()) for i in top)}"
                         f"   rank(hidden) {', '.join(str(r[k]) for k in miss)}")
print("\n".join(lines))
if a.out:
    open(a.out, "w").write("\n".join(lines) + "\n")
