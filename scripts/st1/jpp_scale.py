"""st1 check: scale of J++ outputs vs the final norm's eps; logits of raw J++ vs unit(J++) on FIT rows."""
import sys, torch
sys.path.insert(0, "/workspace/o-lens-improvement/src"); sys.path.insert(0, "/workspace/o-lens-improvement/scripts/perf1")
from olens.readout import Bench
from workspace_lens.utils import load_lens_file
from lite_model import get_lite_model
torch.set_grad_enabled(False)
R = "/workspace/o-lens-improvement"
M, bench, jpp = sys.argv[1], sys.argv[2], sys.argv[3]
model = get_lite_model(M)
norm = model._final_norm
print("norm", type(norm).__name__, "eps", getattr(norm, "variance_epsilon", getattr(norm, "eps", None)))
B = Bench.load(f"{R}/{bench}")
J = {int(l): v.float().cuda() for l, v in load_lens_file(f"{R}/{jpp}", hf_model_name=M).jacobians_L_dict_FN.items()}
fi = torch.tensor(B.index("fit"))
for l in B.layers:
    h = B.resid[l][fi].cuda()
    v = h @ J[l].T
    rms = v.pow(2).mean(-1).sqrt()
    u = v / rms[:, None]
    a, b = model.unembed(v).float(), model.unembed(u).float()
    t = (a.topk(10, -1).indices == b.topk(10, -1).indices).all(-1).float().mean()
    print(f"L{l}: J++ out rms median {rms.median():.2e} min {rms.min():.2e}; resid rms {h.pow(2).mean(-1).sqrt().median():.2e}; "
          f"top-10 identical raw vs unit {t:.3f}; max |logit diff| {(a - b).abs().max():.2f}", flush=True)
