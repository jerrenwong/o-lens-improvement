"""th1 extra (CPU): share of THINK class-offset energy inside the span of the 200 SAY (J++) directions, vs the span of
200 random directions and vs the span of 200 g*u_X directions; and share of centred test-activation variance there."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from th1_common import R, BIG
from workspace_lens.utils import load_lens_file
torch.set_grad_enabled(False)
D = torch.load(f"{BIG}/acts.pt", weights_only=False)
X, wi = D["X"], D["wi"]; NW = 200
tr = torch.tensor([s == "train" for s in D["split"]]); te = ~tr
g, uX = D["g"], D["uX"]
JPP = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN
out = {}
gen = torch.Generator().manual_seed(0)
for L in range(0, 31, 3):
    h = X[:, L].float(); mu = h[tr].mean(0); hc = h - mu
    A = torch.zeros(NW, h.shape[1]).index_add_(0, wi[tr], hc[tr]) / torch.bincount(wi[tr], minlength=NW)[:, None]
    S = (g * uX) @ JPP[L].float()
    res = {}
    for name, B in (("SAY", S - S.mean(0)), ("UNEMB", g * uX - (g * uX).mean(0)), ("RAND", torch.randn(NW, h.shape[1], generator=gen))):
        Q, _ = torch.linalg.qr(B.T)  # [d,200]
        res[name] = dict(offset_energy=round(float((A @ Q).pow(2).sum() / A.pow(2).sum()), 4),
                         test_act_var=round(float((hc[te] @ Q).pow(2).sum() / hc[te].pow(2).sum()), 4))
    out[L] = res
    print(L, res, flush=True)
json.dump(out, open(f"{BIG}/span.json", "w"), indent=1)
