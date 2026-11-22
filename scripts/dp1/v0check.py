"""dp1 V0 reproduction: relative Frobenius difference per layer between our V0 and the stored lens of the same recipe.
    python scripts/dp1/v0check.py <set>   (2b_plain | 9b_plain | 2b_jpp); writes runs/dp1/<set>/v0_check.json"""
import json
import os
import sys

sys.path.insert(0, "/root/jpp_lens/src")
import torch as t  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402
from workspace_lens.fitting.condense_experts import ExpertJacobians  # noqa: E402

R = "/workspace/o-lens-improvement"
s = {"plain2b": "2b_plain", "plain9b": "9b_plain", "jpp2b": "2b_jpp"}.get(sys.argv[1], sys.argv[1])
d = f"{R}/runs/dp1/{s}"
M = "Qwen/Qwen3.5-9B" if s.startswith("9b") else "Qwen/Qwen3.5-2B"


def lens(p):
    return {int(k): v.float() for k, v in load_lens_file(p, hf_model_name=M).jacobians_L_dict_FN.items()}


def rel(a, b):
    return {l: round(float((a[l] - b[l]).norm() / b[l].norm()), 5) for l in a if l in b}


out = {}
if s == "9b_plain":
    st = lens(f"{R}/runs/v6_9b/jlens9b.pt")
    out["V0_at48_vs_stored(48 prompts)"] = rel(lens(f"{d}/V0at48_pooled.pt"), st)
    out["V0_64_vs_stored(48 prompts)"] = rel(lens(f"{d}/var/V0/pooled.pt"), st)
elif s == "2b_plain":
    out["V0_vs_released_jlens(other recipe)"] = rel(lens(f"{d}/var/V0/pooled.pt"), lens(f"{R}/artifacts/baselines/jlens_prefit.pt"))
elif s == "2b_jpp":
    a = ExpertJacobians.load(f"{d}/var/V0/experts_K8.pt").experts_L_dict_EFN
    b = ExpertJacobians.load(f"{R}/runs/r1_2b/A/experts_K8.pt").experts_L_dict_EFN
    out["V0_experts_vs_r1A_experts"] = {l: round(float((a[l] - b[l]).norm() / b[l].norm()), 5) for l in a}
    out["V0_jpp_vs_r1A_jpp"] = rel(lens(f"{d}/var/V0/jpp.pt"), lens(f"{R}/runs/r1_2b/A/jpp_A.pt"))
    out["V0_pooled_vs_r1A_pooled"] = rel(lens(f"{d}/var/V0/pooled.pt"), lens(f"{R}/runs/r1_2b/A/pooled_A.pt"))
    out["V0_jpp_vs_released_jpp"] = rel(lens(f"{d}/var/V0/jpp.pt"), lens(f"{R}/artifacts/baselines/jpp_prefit.pt"))
json.dump(out, open(f"{d}/v0_check.json", "w"))
print(json.dumps(out))
