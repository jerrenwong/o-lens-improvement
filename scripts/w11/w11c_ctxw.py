"""w11c: per-context weight under W1 (tr C_c), W2 (tr C2_c), W0 (tr C_c + |ubar_c|^2) over the M3 contexts (forward only)
    python scripts/w11/w11c_ctxw.py 877 -> runs/w11/m3_ctxweights_2b.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import SKIP, make_windows  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
import copy  # noqa: E402

N = int(sys.argv[1])
m = get_hf_model("Qwen/Qwen3.5-2B", attn_implementation="sdpa")
W = m._lm_head.weight.float(); un2 = (W ** 2).sum(1)
norm = copy.deepcopy(m._final_norm).float()
wins, _ = make_windows(m.tokenizer, 128, 3000)
st = []
with torch.no_grad():
    for i in range(0, N, 16):
        ids = torch.tensor(wins[i:min(i + 16, N)], device="cuda")
        with ActivationRecorder(m.layers, at=[m.n_layers - 1]) as rec:
            m.forward(ids)
        for j in range(ids.shape[0]):
            p = torch.softmax(norm(rec.activations[m.n_layers - 1][j].float()) @ W.T, -1)[SKIP:-1]
            ub = p @ W; q = p ** 2; wq = q @ W
            trC = (p * un2).sum(1) - (ub ** 2).sum(1)
            trC2 = (q * un2).sum(1) - 2 * (ub * wq).sum(1) + q.sum(1) * (ub ** 2).sum(1)
            st.append(torch.stack([trC, trC2, (ub ** 2).sum(1), p.max(1).values], 1).cpu())
S = torch.cat(st).double()
qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
names = ["trC (W1 weight)", "trC2 (W2 weight)", "|ubar|^2", "max p"]
out = dict(n_ctx=int(S.shape[0]), quantiles_10_25_50_75_90={k: [round(float(x), 4) for x in torch.quantile(S[:, j], qs)] for j, k in enumerate(names)})
for thr in (0.5, 0.9):
    c = S[:, 3] > thr
    out[f"maxp>{thr}"] = dict(share_of_contexts=float(c.double().mean()),
                              share_of_W1_weight=float(S[c, 0].sum() / S[:, 0].sum()),
                              share_of_W2_weight=float(S[c, 1].sum() / S[:, 1].sum()),
                              share_of_W0_weight=float((S[c, 0] + S[c, 2]).sum() / (S[:, 0] + S[:, 2]).sum()))
out["ubar_share_of_W0_trace"] = float(S[:, 2].sum() / (S[:, 0] + S[:, 2]).sum())
(R / "runs/w11/m3_ctxweights_2b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
