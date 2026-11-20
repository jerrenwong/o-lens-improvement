"""r1 diagnostics per lens (Qwen3.5-2B): share of filtered top-10 slots starting with '@' on the association
prompts at each readout layer (FIT+DEV items of auto/split_2b.json; test items never loaded into any statistic),
as in auto/q7_fm2b.py section 4a; and the decoded top-3 singular output directions (both signs) at layers 15/18/21.

    cd /workspace/o-lens-improvement && /root/jpp_lens/.venv/bin/python scripts/r1_diag.py name=path.pt ... > log
"""
import json
import sys

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from olens.readout import Bench  # noqa: E402

R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-2B"; LAYERS = [3, 6, 9, 12, 15, 18, 21]
t.set_grad_enabled(False)
lenses = dict(a.split("=", 1) for a in sys.argv[1:])
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-2b.pt")
sp = json.load(open(f"{R}/auto/split_2b.json"))
test = {tuple(x) for x in sp["test"]}
keep = {tuple(x) for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in keep and b.eval_slugs[i] == "association"]
assert all((b.eval_slugs[i], b.item_names[i]) not in test for i in idx)
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()
NEG = float("-inf")
print(f"association fit+dev prompts: {len(idx)}")
out = {}
for name, path in lenses.items():
    J = load_lens_file(path, hf_model_name=M).jacobians_L_dict_FN
    row = {}
    for L in LAYERS:
        v = b.resid[L][idx].cuda().float() @ J[L].cuda().float().T
        tp = model.unembed(v).float().masked_fill(punct, NEG).topk(10, -1).indices
        row[L] = float(at[tp].float().mean())
    out[name] = row
    print(f"@share {name:12s} " + " ".join(f"L{L} {row[L]:.2f}" for L in LAYERS), flush=True)
    for L in (15, 18, 21):
        U, S, _ = t.linalg.svd(J[L].cuda().float(), full_matrices=False)
        def dec(u):
            return " ".join(strs[j].strip() for j in model.unembed(u[None]).float().masked_fill(punct, NEG).topk(6, -1).indices[0].tolist())
        print(f"  {name} L{L} sigma1..3/median {[round(float(x / S.median()), 1) for x in S[:3]]}")
        for k in range(3):
            print(f"     u{k+1}+: {dec(U[:, k])}   |   u{k+1}-: {dec(-U[:, k])}")
    print(flush=True)
json.dump(out, open(f"{R}/runs/r1_2b/diag_at_share.json", "w"), indent=1)
