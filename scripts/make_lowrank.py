"""Rank-r truncations of a linear lens: J_l -> U_r S_r V_r^T per layer (SVD on GPU).

With the random-projection Jacobian loss, E||(G - J_r)^T u||^2 = E||G - J||_F^2 + ||J - J_r||_F^2
(J = E[G]), so the truncated SVD of J is exactly the best rank-r Jacobian-matching lens.

    python scripts/make_lowrank.py <lens.pt> <out_dir> 16,64,256,1024 [name_prefix=lr]
Writes <out_dir>/<prefix><r>.pt (olens.TrainedLens, fp32) and lowrank_energy.json (kept share of ||J||_F^2).
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch as t
from workspace_lens.utils import load_lens_file

import olens  # noqa: F401
from olens.lens import TrainedLens

src, out, ranks = sys.argv[1], Path(sys.argv[2]), [int(r) for r in sys.argv[3].split(",")]
prefix = sys.argv[4] if len(sys.argv) > 4 else "lr"
out.mkdir(parents=True, exist_ok=True)
lens = load_lens_file(src, hf_model_name="Qwen/Qwen3.5-2B")
layers = lens.source_layers
svd = {l: t.linalg.svd(lens.jacobians_L_dict_FN[l].cuda().float()) for l in layers}
energy = {}
for r in ranks:
    Js, kept = {}, []
    for l in layers:
        U, S, Vh = svd[l]
        Js[l] = ((U[:, :r] * S[:r]) @ Vh[:r]).cpu()
        kept.append(float((S[:r] ** 2).sum() / (S**2).sum()))
    cfg = replace(lens.config, checkpoint_name=f"{prefix}{r}", lens_type="trained")  # type: ignore[arg-type]
    TrainedLens(cfg, jacobians=Js).save(str(out / f"{prefix}{r}.pt"), dtype=t.float32)
    energy[r] = {"mean_kept_frobenius_share": sum(kept) / len(kept), "per_layer": kept}
    print(f"rank {r}: kept {energy[r]['mean_kept_frobenius_share']:.3f} of ||J||_F^2 (mean over layers)")
(out / f"{prefix}_energy.json").write_text(json.dumps(energy, indent=1))
