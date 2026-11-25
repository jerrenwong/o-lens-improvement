"""w11c: M3 estimator stability. For lam in LAMS, M_lam = (B + lam s I)^-1 (A + lam s Jz), s = tr(B)/d:
 (1) split-half: relative Frobenius distance between the two halves' M (even / odd windows of the m3x pass) for the
     exact top-10 p^2 (W2x), p^3 (W3x) and the sampled p^2 (W2s, r = 8) estimators;
 (2) W2-exact (m3x pass) vs W2-sampled (m3b pass, r = 16, different contexts) on the full data.
Distances are on M - Jz relative to |M_W2x - Jz| where noted ("delta rel"), and on M itself ("relF").
    python scripts/w11/w11c_split.py  -> runs/w11/m3_split_2b.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
from workspace_lens.utils import load_lens_file  # noqa: E402
sys.path += [str(R / "src"), str(R / "scripts/perf1")]
from lite_model import get_lite_model  # noqa: E402

LAMS = [10.0, 1.0, 0.1, 0.01]
m = get_lite_model("Qwen/Qwen3.5-2B")
norm, head = m._final_norm, m._lm_head
p = next(norm.parameters())
gam = norm(torch.ones(1, head.weight.shape[1], device=p.device, dtype=torch.float32)).float()[0].double()
J = {int(l): v.cuda().double() for l, v in load_lens_file(str(R / "artifacts/baselines/jlens_prefit.pt"),
                                                         hf_model_name="Qwen/Qwen3.5-2B").jacobians_L_dict_FN.items()}
x = torch.load(R / ".hf27/w11/m3x_2b.pt", map_location="cpu")
b = torch.load(R / ".hf27/w11/m3b_2b.pt", map_location="cpu")
rho = b["rho"]
L = x["layers"]
I = torch.eye(gam.shape[0], device="cuda", dtype=torch.float64)


def solve(A, B, Jz, lam):
    s = float(torch.trace(B)) / B.shape[0]
    return torch.linalg.solve(B + lam * s * I, A + lam * s * Jz)


rel = lambda a, c: float((a - c).norm() / c.norm())  # noqa: E731
out = {}
for lam in LAMS:
    o = {}
    for l in L:
        Jz = rho * gam[:, None] * J[l]
        e = {}
        for name, ak, bk in (("W2x", "A2x", "B2x"), ("W3x", "A3x", "B3x"), ("W2s", "A2s", "B2x")):
            Ms = [solve(x[ak][h][l].cuda(), x[bk][h].cuda(), Jz, lam) for h in range(2)]
            e[f"{name} split-half relF"] = round(rel(Ms[0], Ms[1]), 4)
            e[f"{name} split-half delta-cos"] = round(float(torch.nn.functional.cosine_similarity(
                (Ms[0] - Jz).flatten(), (Ms[1] - Jz).flatten(), dim=0)), 4)
        M2x = solve(x["A2x"][0][l].cuda() + x["A2x"][1][l].cuda(), (x["B2x"][0] + x["B2x"][1]).cuda(), Jz, lam)
        M2s = solve(b["A2"][l].cuda(), b["B2"].cuda(), Jz, lam)
        e["W2x vs W2(sampled, m3b) relF"] = round(rel(M2x, M2s), 4)
        e["W2x vs W2(sampled) delta-cos"] = round(float(torch.nn.functional.cosine_similarity(
            (M2x - Jz).flatten(), (M2s - Jz).flatten(), dim=0)), 4)
        e["W2x relF to Jz"] = round(rel(M2x, Jz), 4)
        o[l] = e
    out[str(lam)] = o
    print(lam, json.dumps({l: o[l] for l in (L[0], L[3], L[-1])}), flush=True)
(R / "runs/w11/m3_split_2b.json").write_text(json.dumps(out, indent=1))
