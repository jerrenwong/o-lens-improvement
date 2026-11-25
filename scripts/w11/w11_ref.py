"""w11 (pod): reference activations for z-scoring token scores: 64 windows of the w11 window list NOT used by the
accumulation (indices 1400..1463 of make_windows(1500)), valid positions 16..S-2, 4096 random positions, per readout layer.
    python scripts/w11/w11_ref.py <model> <tag>  -> .hf27/w11/ref_<tag>.pt {layer: [4096, d] fp16}"""
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from w11_accum import LAYERS, SKIP, make_windows  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

name, tag = sys.argv[1], sys.argv[2]
m = get_hf_model(name, attn_implementation="sdpa")
wins, _ = make_windows(m.tokenizer, 128, 1500)
ids = torch.tensor(wins[1400:1464], device="cuda")
L = LAYERS[name]
out = {l: [] for l in L}
with torch.no_grad():
    for i in range(0, 64, 8):
        with ActivationRecorder(m.layers, at=L) as rec:
            m.forward(ids[i:i + 8])
        for l in L:
            out[l].append(rec.activations[l][:, SKIP:-1].reshape(-1, m.d_model).float().cpu())
g = torch.Generator().manual_seed(0)
X = {l: torch.cat(v) for l, v in out.items()}
sel = torch.randperm(X[L[0]].shape[0], generator=g)[:4096]
torch.save({l: v[sel].half() for l, v in X.items()}, R / f".hf27/w11/ref_{tag}.pt")
print("saved", {l: tuple(v[sel].shape) for l, v in X.items()})
