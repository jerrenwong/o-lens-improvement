"""perf1 Part B: numerical drift between kernel modes on the same 2,000 al2 sentences (last-token residuals at
layers 4..28) and its effect on the released J++ lens's top-10 (decoded as scripts/mp1/mp1_fit.py: logits =
((J h) * g).bf16 @ U.T, non-alphanumeric tokens masked). Also compares generated texts across modes.
Pairs that are missing (mode not run) are skipped.   python scripts/perf1/analyze_b.py
"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
D = R / ".hf27" / "perf1"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
sys.path.insert(0, str(R / "src"))
from workspace_lens.utils import load_lens_file  # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda"


def load(name):
    p = D / f"{name}.pt"
    if not p.exists():
        return None
    return {int(l): v.to(dev).float() for l, v in torch.load(p).items()}


meta = torch.load(R / ".hf27/al2/meta.pt", weights_only=False)
g = meta["g"].to(dev)
U = torch.load(R / ".hf27/al1/U.pt").to(dev)
strs = torch.load(R / ".hf27/al1/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
J = {int(l): v.to(dev).float() for l, v in
     load_lens_file(str(R / "artifacts/baselines/jpp9b_prefit.pt"), hf_model_name="Qwen/Qwen3.5-9B")
     .jacobians_L_dict_FN.items()}


def top10(h, layer):
    out = []
    for i in range(0, h.shape[0], 500):
        v = h[i:i + 500] @ J[layer].T
        lg = ((v * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        out.append(lg.topk(10).indices)
    return torch.cat(out)


stored = {l: torch.load(R / f".hf27/al2/X_L{l}.pt")[:2000].to(dev).float() for l in LAYERS}
runs = {n: load(n) for n in ["resid_torch_bs128", "resid_torch_bs32", "resid_fla_bs128", "resid_fla_bs32",
                             "resid_fla_cc_bs128", "resid_fla_cc_bs32"]}
runs["stored_al2_X_fp16"] = stored
PAIRS = [("resid_fla_bs128", "resid_fla_bs32", "current (fla) run-to-run: batch 128 vs 32"),
         ("resid_torch_bs128", "resid_torch_bs32", "pure torch run-to-run: batch 128 vs 32"),
         ("resid_fla_bs128", "stored_al2_X_fp16", "current (fla) vs stored al2 X_L* (fp16, other batches)"),
         ("resid_fla_bs128", "resid_fla_cc_bs128", "current (fla) vs fla + causal-conv1d"),
         ("resid_fla_cc_bs128", "resid_fla_cc_bs32", "fla + causal-conv1d run-to-run: batch 128 vs 32"),
         ("resid_fla_bs128", "resid_torch_bs128", "current (fla) vs pure torch")]
T = {}
res = {}
for a, b, label in PAIRS:
    A, B = runs.get(a), runs.get(b)
    if A is None or B is None:
        continue
    per = {}
    for l in LAYERS:
        x, y = A[l], B[l]
        rel = (x - y).norm(dim=-1) / x.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        for n, H in ((a, x), (b, y)):
            if (n, l) not in T:
                T[(n, l)] = top10(H, l)
        ta, tb = T[(a, l)], T[(b, l)]
        set_changed = int((ta.sort(-1).values != tb.sort(-1).values).any(-1).sum())
        order_changed = int((ta != tb).any(-1).sum())
        top1_changed = int((ta[:, 0] != tb[:, 0]).sum())
        per[l] = dict(rel_l2_max=rel.max().item(), rel_l2_mean=rel.mean().item(), cos_min=cos.min().item(),
                      cos_mean=cos.mean().item(), bitwise_equal_rows=int((x == y).all(-1).sum()),
                      jpp_top10_set_changed=set_changed, jpp_top10_order_changed=order_changed,
                      jpp_top1_changed=top1_changed)
    agg = dict(rel_l2_max=max(p["rel_l2_max"] for p in per.values()),
               rel_l2_mean=sum(p["rel_l2_mean"] for p in per.values()) / len(per),
               cos_min=min(p["cos_min"] for p in per.values()),
               jpp_top10_set_changed_sum=sum(p["jpp_top10_set_changed"] for p in per.values()),
               jpp_top10_set_changed_max_layer=max(p["jpp_top10_set_changed"] for p in per.values()),
               jpp_top1_changed_sum=sum(p["jpp_top1_changed"] for p in per.values()),
               rows_x_layers=2000 * len(LAYERS))
    res[label] = dict(a=a, b=b, all_layers=agg, per_layer=per)
    print(f"{label}: {json.dumps(agg)}", flush=True)

gens = {}
for m in ("torch", "fla", "fla_cc"):
    p = D / f"gen_{m}.json"
    if p.exists():
        gens[m] = json.load(open(p))
for a, b in (("fla", "torch"), ("fla", "fla_cc")):
    if a in gens and b in gens:
        ga, gb = gens[a], gens[b]
        same_s = sum(x == y for x, y in zip(ga["sample"], gb["sample"]))
        same_g = sum(ga["greedy"][k] == gb["greedy"].get(k) for k in ga["greedy"])
        res[f"gen {a} vs {b}"] = dict(sampled_identical=f"{same_s}/{len(ga['sample'])}",
                                      greedy_identical=f"{same_g}/{len(ga['greedy'])}")
        print(f"gen {a} vs {b}: {res[f'gen {a} vs {b}']}", flush=True)
(R / "scripts/perf1/results/drift_b.json").write_text(json.dumps(res, indent=1))
print("ANALYZE_DONE")
