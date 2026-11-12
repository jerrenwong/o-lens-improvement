"""Held-out reconstruction (Jacobian rel. error, KL to final output) for saved lens files.

Same windows / fixed u / positions as olens.train's held-out eval (paragraph-start WikiText
validation, positions >= --skip-first), so numbers are comparable with v2's train log.

    python scripts/heldout_eval.py --out results/x.json --skip-first 1 a.pt b.pt ...
"""

import argparse
import json
from pathlib import Path

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

import olens  # noqa: F401
from olens.train import Arm, heldout_metrics, wikitext_windows

ap = argparse.ArgumentParser()
ap.add_argument("lenses", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--skip-first", type=int, default=1)
ap.add_argument("--val-windows", type=int, default=64)
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
args = ap.parse_args()

model = get_hf_model(args.model, attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
tgt = len(model.layers) - 1
src = list(range(tgt))
val = wikitext_windows(model.tokenizer, "validation", args.val_windows, 128, Path("/root/cache"), para_start=True)


def stack(d):
    return None if d is None else t.stack([d[l] for l in src]).cuda().float()


lenses = {}
for path in args.lenses:
    L = load_lens_file(path, hf_model_name=args.model)
    lenses[Path(path).stem] = Arm(
        name=Path(path).stem, objective="eval", m=0, J=stack(L.jacobians_L_dict_FN),
        w_in=stack(getattr(L, "w_in_L_dict_MN", None)), w_out=stack(getattr(L, "w_out_L_dict_FM", None)),
        bias=stack(getattr(L, "bias_L_dict_F", None)),
    )
res = heldout_metrics(model, lenses, val, src, tgt, 16, 12345, args.skip_first)
summary = {k: {"jac_rel_err": sum(v["jac_rel_err"]) / len(src), "kl": sum(v["kl"]) / len(src)} for k, v in res.items()}
for k, v in summary.items():
    print(f"{k:16s} jac_rel_err {v['jac_rel_err']:.4f}  KL {v['kl']:.3f}")
Path(args.out).write_text(json.dumps({"summary": summary, "per_layer": res}, indent=1))
