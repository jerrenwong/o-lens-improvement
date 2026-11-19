#!/bin/bash
# ~1-minute end-to-end check of olens.bsf_train on a GPU pod.
# Usage (on the pod): bash /workspace/o-lens-improvement/scripts/test_bsf_train.sh
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
OUT=${OUT:-/root/bsf_test}
rm -rf "$OUT"
cd "$R"
$PY -m olens.bsf_train --jlens artifacts/baselines/jlens_prefit.pt \
  --tokens 200000 --checkpoints 100000,200000 --latents 4096 --group-sizes 1,4 \
  --val-windows 16 --dead-tokens 50000 --warmup-steps 5 --out "$OUT"
ls "$OUT"
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
import torch as t
from olens.bsf_io import BlockDict
from olens.bsf_train import to_space
from workspace_lens.utils import load_lens_file

out = Path(sys.argv[1])
args = json.loads((out / "args.json").read_text())
J = load_lens_file(args["jlens"], hf_model_name=args["model"]).jacobians_L_dict_FN[18].cuda().float()
h = t.load(out / "val_acts_L18.pt").cuda()
print("val acts", tuple(h.shape), h.dtype)
for name in ["L18_resid_b4_0.2M", "L18_jspace_b1_0.2M"]:
    dd = BlockDict.load(out / f"{name}.pt", device="cuda")
    x = to_space(h, dd.space, J)
    with t.no_grad():
        z, blocks, err = dd.split(x)
    gap = (x - (dd.mean + blocks + err)).abs().max().item()
    nnz = (z.norm(dim=-1) > 0).sum(-1)
    fvu = (err.pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
    print(f"{name}: identity gap {gap:.2e}, blocks/row min {nnz.min().item()} max {nnz.max().item()} "
          f"(l0={dd.l0}), fvu {fvu:.4f} (saved {dd.meta['fvu']:.4f})")
    assert gap < 1e-3 * x.abs().max().item(), "x != mean + blocks + err"
    assert (nnz == dd.l0).all(), "not exactly l0 blocks per row"
    st = t.load(out / f"{name}_stats.pt", weights_only=True)
    print("  stats", {k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in st.items()})
print("OK")
EOF
