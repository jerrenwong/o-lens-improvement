"""w11b: mean activation mu_l over the accumulation contexts (valid source positions 16..126 of the first N windows of
make_windows, N = the M1 pass's window count), all 7 readout layers; also mean |h_l|.
    python scripts/w11/w11b_mu.py 2737  -> .hf27/w11/mu_2b.pt {mu: {l: [d]}, mean_norm: {l}, n}"""
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import LAYERS, SKIP, make_windows  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

N = int(sys.argv[1]); name = "Qwen/Qwen3.5-2B"
m = get_hf_model(name, attn_implementation="sdpa")
L = LAYERS[name]
wins, _ = make_windows(m.tokenizer, 128, 4500)
wins = wins[:N]
s = {l: torch.zeros(m.d_model, dtype=torch.float64, device="cuda") for l in L}
nn = {l: 0.0 for l in L}
n = 0
with torch.no_grad():
    for i in range(0, N, 32):
        ids = torch.tensor(wins[i:i + 32], device="cuda")
        with ActivationRecorder(m.layers, at=L) as rec:
            m.forward(ids)
        for l in L:
            h = rec.activations[l][:, SKIP:-1].float()
            s[l] += h.sum((0, 1)).double()
            nn[l] += float(h.norm(dim=-1).sum())
        n += ids.shape[0] * (ids.shape[1] - 1 - SKIP)
mu = {l: (v / n).float().cpu() for l, v in s.items()}
mn = {l: nn[l] / n for l in L}
torch.save(dict(mu=mu, mean_norm=mn, n=n), R / ".hf27/w11/mu_2b.pt")
print({l: round(float(mu[l].norm()) / mn[l], 3) for l in L}, "n", n)
