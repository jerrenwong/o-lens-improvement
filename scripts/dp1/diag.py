"""dp1 diagnostic on the FIT split items only (fit:0, the expert-weight fold; = auto/split_*.json "fit").

    python scripts/dp1/diag.py --model Qwen/Qwen3.5-2B --out OUT.json name=lens.pt [name=lens.pt ...]

Per lens and readout layer: share of fit items whose lens top-1 token (unfiltered; and with Readout Filtering)
equals the model's own next token at the readout position (one forward pass per fit prompt), the share of the
lens's top-10 (unfiltered) that is non-content (no letter, or among the 200 most frequent fit-pool tokens:
runs/t1_2b/content_rule.json), and tr(J)/d. Residuals = the bench's fit rows (checked against this forward).
"""
import argparse
import json
import os
import sys

R = "/workspace/o-lens-improvement"
JPP = "/root/jpp_lens"
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("lenses", nargs="+")
args = ap.parse_args()
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
sys.path.insert(0, os.path.join(R, "src"))
import torch as t  # noqa: E402
import jpp_cli  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402
from olens.readout import Bench  # noqa: E402

tag = {"Qwen/Qwen3.5-2B": "2b", "Qwen/Qwen3.5-9B": "9b"}[args.model]
layers = {"2b": [3, 6, 9, 12, 15, 18, 21], "9b": [4, 8, 12, 16, 20, 24, 28]}[tag]
model = get_hf_model(args.model, attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
max_len = jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN
_, fit_items, _ = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=READOUT_EVALS_DATA_DIR, correctness_csv=f"{R}/data/model_correctness.csv",
    hf_model_name=args.model, layers=layers, readout_max_seq_len=max_len, items_spec="fit:0", fit_fraction=0.2)
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-{tag}.pt")
bidx = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
fit_b = bench.index("fit")
assert sorted(bidx[(it.eval_slug, it.name)] for it in fit_items) == fit_b, "fit items differ from bench fit split"
rule = json.load(open(f"{R}/runs/t1_2b/content_rule.json"))
content = t.tensor(rule["content"], dtype=t.bool)
excl = bench.excluded_ids

target = model.n_layers - 1
nxt, rows, maxdev = [], [], 0.0
with t.no_grad():
    for it in fit_items:
        ro = re_mod._item_readout_and_candidates(model, it, max_len)
        with ActivationRecorder(model.layers, at=[target, layers[0]]) as rec:
            model.forward(ro.input_ids_Int_1S)
        pos = ro.readout_position
        logits = model.unembed(rec.activations[target][0, pos][None].float())[0].float()
        nxt.append(int(logits.argmax()))
        b = bidx[(it.eval_slug, it.name)]
        rows.append(b)
        maxdev = max(maxdev, float((rec.activations[layers[0]][0, pos].float().cpu() - bench.resid[layers[0]][b]).abs().max()
                                   / bench.resid[layers[0]][b].abs().max()))
nxt_t = t.tensor(nxt)
print(f"fit items {len(rows)}; bench-resid check max rel dev {maxdev:.2e}; next tokens non-content share "
      f"{float((~content[nxt_t]).float().mean()):.3f}", flush=True)
out = {"n_fit_items": len(rows), "resid_check_max_rel_dev": maxdev,
       "next_token_noncontent_share": float((~content[nxt_t]).float().mean()), "lenses": {}}
ri = t.tensor(rows)
for spec in args.lenses:
    name, path = spec.split("=", 1)
    L = load_lens_file(path, hf_model_name=args.model)
    res = {}
    for l in layers:
        J = L.jacobians_L_dict_FN[l].float().cuda()
        h = bench.resid[l][ri].cuda()
        v = h @ J.T
        lg = model.unembed(v).float()
        top10 = lg.topk(10, dim=-1).indices.cpu()
        lgf = lg.clone()
        lgf[:, excl.cuda()] = -float("inf")
        top1f = lgf.argmax(-1).cpu()
        res[l] = {
            "top1_eq_next": float((top10[:, 0] == nxt_t).float().mean()),
            "top1f_eq_next": float((top1f == nxt_t).float().mean()),
            "next_in_top10": float((top10 == nxt_t[:, None]).any(1).float().mean()),
            "top10_noncontent_share": float((~content[top10.clamp(max=content.numel() - 1)]).float().mean()),
            "tr_over_d": float(t.trace(J) / J.shape[0]),
        }
    out["lenses"][name] = res
    print(name, json.dumps(res), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
