"""q2: the constant ("prior") part of each A-lens map. For each map (w9 format: out = (h - mx) W + my) and readout
layer, on the val passages (rows 0..999 of the passage set; never test rows):
  const share = |my|^2 / mean |out|^2  (D3's "constant share of |out|^2"; mx = training mean, so my = the mean output
  over training passages), and the top-10 tokens of the constant alone (unembed(my), f4-style word tokens only).
    python q2_prior.py <model> <map stem> [<map stem> ...]   -> runs/q2/prior_<tag>.json (+ printed table)"""
import json
import sys

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/perf1")
sys.path.insert(0, f"{R}/scripts/st1")
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

M = sys.argv[1]
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
torch.set_grad_enabled(False)
model = get_lite_model(M)
C.setup(model)
X = torch.load(XFILE, map_location="cpu")["X"]
res = {}
for stem in sys.argv[2:]:
    p = next(f"{d}/{stem}.pt" for d in (f"{R}/.hf27/q2/maps", f"{R}/.hf27/w9/maps")
             if __import__("os").path.exists(f"{d}/{stem}.pt"))
    m = torch.load(p, map_location="cpu")
    res[stem] = {}
    for l in LAYERS:
        v = m[l]
        mx, W, my = v["mx"].cuda().float(), v["W"].cuda().float() * v["scale"], v["my"].cuda().float()
        h = X[l][:1000].cuda().float()
        out = (h - mx) @ W + my
        share = (my.pow(2).sum() / out.pow(2).sum(-1).mean()).item()
        top = C.topk(my[None], 10)[0]
        res[stem][l] = dict(const_share=share, const_top=[C.strs[int(i)] for i in top])
    mid = LAYERS[len(LAYERS) // 2]
    print(f"{stem}: const share by layer " + ", ".join(f"L{l} {res[stem][l]['const_share']:.2f}" for l in LAYERS)
          + f" | constant's top-10 @L{mid}: {res[stem][mid]['const_top']}", flush=True)
json.dump(res, open(f"{R}/runs/q2/prior_{TAG}.json", "w"), ensure_ascii=False, indent=1)
