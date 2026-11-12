"""Relative Frobenius distance between two linear lenses, per layer."""
import sys, torch as t
from workspace_lens.utils import load_lens_file
import olens  # noqa
a, b = (load_lens_file(p, hf_model_name="Qwen/Qwen3.5-2B") for p in sys.argv[1:3])
ds = []
for l in range(23):
    A, B = a.jacobians_L_dict_FN[l].float(), b.jacobians_L_dict_FN[l].float()
    ds.append(((A - B).norm() / B.norm()).item())
print("rel ||A-B||/||B|| per layer:", " ".join(f"{d:.3f}" for d in ds))
print(f"mean {sum(ds)/len(ds):.4f}  max {max(ds):.4f}")
