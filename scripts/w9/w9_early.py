"""w9 reader check (c), step 1: residuals of every DEV item at the benchmark's readout token and at 3 tokens earlier.
Items, tokenisation and readout positions exactly as scripts/bench_cache.py (jpp_cli / readout_evals helpers, same
correctness csv and max_seq_len as stored in the bench file). Check: residual at the readout token vs the stored bench
residual (cosine). Output: .hf27/w9/early_<tag>.pt {idx (bench indices), pos, off: {0: {layer: [n,d]}, 3: {...}}, cos}
Usage: python w9_early.py <model>"""
import os
import sys

import torch

R = "/workspace/o-lens-improvement"
JPP = "/root/jpp_lens"
M = sys.argv[1]
TAG = "2b" if "2B" in M else "9b"
OFFS = [0, 3]
sys.path.insert(0, f"{R}/src")
from olens.readout import Bench  # noqa: E402
import json  # noqa: E402

bpath = f"{R}/artifacts/bench/qwen3.5-{TAG}.pt"
raw = torch.load(bpath, map_location="cpu", weights_only=True)
bench = Bench.load(bpath)
split = json.load(open(f"{R}/auto/split_{TAG}.json"))
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import jpp_cli  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

torch.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa")
layers = list(range(model.n_layers - 1))
msl = int(raw["max_seq_len"])
_, scoreable, _ = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=READOUT_EVALS_DATA_DIR, correctness_csv=raw["correctness_csv"], hf_model_name=M,
    layers=layers, readout_max_seq_len=msl, items_spec="all", fit_fraction=float(raw["fit_fraction"]))
by_key = {(it.eval_slug, it.name): it for it in scoreable}
pos_b = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
dev_idx = sorted(pos_b[tuple(k)] for k in split["dev"])
RL = sorted(bench.resid.keys())
out = {o: {l: [] for l in RL} for o in OFFS}
positions, cos = [], []
for i in dev_idx:
    it = by_key[(bench.eval_slugs[i], bench.item_names[i])]
    ro = re_mod._item_readout_and_candidates(model, it, msl)
    assert ro.candidate_ids_by_intermediate == bench.candidates[i]
    p = int(ro.readout_position)
    if bench.readout_positions is not None:
        assert p == bench.readout_positions[i], (p, bench.readout_positions[i])
    rec = {}
    hk = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, inp, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[0]))(l)) for l in RL]
    try:
        model.forward(ro.input_ids_Int_1S.to("cuda"))
    finally:
        for h in hk:
            h.remove()
    for o in OFFS:
        for l in RL:
            out[o][l].append(rec[l][max(p - o, 0)].float().cpu())
    cos.append(min(torch.cosine_similarity(rec[l][p].float().cpu(), bench.resid[l][i], 0).item() for l in RL))
    positions.append(p)
res = dict(idx=dev_idx, pos=positions, off={o: {l: torch.stack(v) for l, v in d.items()} for o, d in out.items()},
           cos=cos)
torch.save(res, f"{R}/.hf27/w9/early_{TAG}.pt")
print(f"[w9 early] {M}: {len(dev_idx)} dev items; readout-token residual vs bench: min cos {min(cos):.5f}, mean "
      f"{sum(cos) / len(cos):.5f}; readout position < 3 for {sum(p < 3 for p in positions)} items", flush=True)
