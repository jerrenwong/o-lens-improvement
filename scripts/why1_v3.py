"""why1 v3 check (auto/PREREG_v3.md amendment 4; single registered use of v3, 9B only, nothing tuned on v3).

Standalone path reusing auto/evaluate_v3.py's scorer pieces (bench_v3 rows, readout_rows chunk_size=1, pass_at_k).
1. Reproduce V3_jpp_9b and V3_e2_A_9b (pass@10/1/unf and every per-row rank vs their CSVs); abort if not exact.
2. T   = unit(J++ h) + 0.2 * unit(bag map (c)(mean input embedding of prompt tokens[:readout_pos+1]))  -> V3_T_9b
   T05 = same with 0.5                                                                                -> V3_T05_9b
Token ids: tok(prompt, add_special_tokens=False); the readout position comes from the bench's
readout_positions and is verified per item by cos(hidden_states[l+1][pos], cached resid) at L12 and L24.
"""
import csv
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "src"))
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

from olens.readout import Bench, readout_rows, write_csv  # noqa: E402
sys.path += [str(R / "auto"), str(R / "auto/candidates")]
from evaluate_v3 import pass_at_k  # noqa: E402
from e2_lib import ELens, load_maps  # noqa: E402

M, LAYERS, dev = "Qwen/Qwen3.5-9B", [4, 8, 12, 16, 20, 24, 28], "cuda"
torch.set_grad_enabled(False)


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


model = get_hf_model(M, attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
tok, hf = model.tokenizer, model._hf_model
EMB = hf.get_input_embeddings().weight
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(str(R / "artifacts/baselines/jpp9b_prefit.pt"), hf_model_name=M).jacobians_L_dict_FN.items()}
EMAPS = load_maps(str(R / "runs/e1/E_9b.pt"), dev)

bench = Bench.load(R / "artifacts/bench/v3_qwen3.5-9b.pt")
split = json.loads((R / "auto/split_v3_9b.json").read_text())
pos_v3 = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
idx = sorted(pos_v3[tuple(k)] for k in split["test"])
manifest = json.loads((R / "data/evals_v3/MANIFEST.json").read_text())
prompts = {}
for fname, meta in manifest["files"].items():
    for it in json.loads((R / "data/evals_v3" / fname).read_text())["items"]:
        prompts[(meta["task"], it["name"])] = it["prompt"]
si = torch.as_tensor(idx, dtype=torch.long)
rr = {l: v[si].to(dev).float() for l, v in bench.resid.items()}

# ---- token ids and readout position check
feats, bad, cmin = [], [], 1.0
for n, i in enumerate(idx):
    ids = tok(prompts[(bench.eval_slugs[i], bench.item_names[i])], add_special_tokens=False)["input_ids"]
    p = bench.readout_positions[i]
    hs = hf(torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
    c = min(torch.cosine_similarity(hs[l + 1][0, p].float(), rr[l][n], 0).item() for l in (12, 24))
    if c < 0.999:
        bad.append((bench.eval_slugs[i], bench.item_names[i], c))
    cmin = min(cmin, c)
    feats.append(EMB[torch.tensor(ids[:p + 1], device=EMB.device)].float().mean(0).to(dev))
print(f"position check: {len(idx)} items, min cos {cmin:.6f}, below .999: {len(bad)} {bad[:5]}", flush=True)
assert not bad, "token ids / readout positions do not reproduce the cached residuals"

mx, W, my = (t.to(dev) for t in torch.load(R / "runs/why1/bagmaps.pt")["c_all"])
bag = (torch.stack(feats) - mx) @ W + my


def score(name, vecs):
    rows = {True: [], False: []}
    for l in LAYERS:
        for f in (True, False):
            rows[f] += readout_rows(name, l, bench, vecs[l], model, filtered=f, items=idx, chunk_size=1)
    s = dict(pass10=pass_at_k(rows[True], 10)[0], pass1=pass_at_k(rows[True], 1)[0],
             pass10_unf=pass_at_k(rows[False], 10)[0], pass1_unf=pass_at_k(rows[False], 1)[0],
             per_task_pass10=pass_at_k(rows[True], 10)[1], n_items=len(idx))
    print(name, json.dumps(s), flush=True)
    return rows, s


def same_as_run(run, rows, s):
    ref = json.loads((R / f"auto/runs/{run}/summary.json").read_text())
    ok = all(abs(ref[k] - s[k]) < 1e-12 for k in ("pass10", "pass1", "pass10_unf", "pass1_unf"))
    for f, kind in ((True, "filtered"), (False, "unfiltered")):
        with open(R / f"auto/runs/{run}/readout_{kind}.csv") as fh:
            refr = {(r["eval"], r["item"], r["intermediate"], int(r["layer"])): int(r["rank"]) for r in csv.DictReader(fh)}
        mine = {(r["eval"], r["item"], r["intermediate"], r["layer"]): r["rank"] for r in rows[f]}
        diff = sum(refr.get(k) != v for k, v in mine.items()) + abs(len(refr) - len(mine))
        print(f"  repro {run} {kind}: {len(mine)} rows, {diff} differ", flush=True)
        ok &= diff == 0
    return ok


jv = {l: rr[l] @ JPP[l].T for l in LAYERS}
okJ = same_as_run("V3_jpp_9b", *score("repro_jpp", jv))
lensA = ELens(JPP, EMAPS, "mix", 0.5)
okA = same_as_run("V3_e2_A_9b", *score("repro_A", {l: lensA.transport(rr[l].clone(), l) for l in LAYERS}))
print(f"REPRO J++ {okJ} A {okA}", flush=True)
assert okJ and okA, "standalone path does not reproduce the v3 scorer exactly"

for name, lam in (("V3_T_9b", 0.2), ("V3_T05_9b", 0.5)):
    rows, s = score(name, {l: unit(jv[l]) + lam * unit(bag) for l in LAYERS})
    out = R / "auto/runs" / name
    write_csv(out / "readout_filtered.csv", rows[True])
    write_csv(out / "readout_unfiltered.csv", rows[False])
    s.update(lens=f"why1 bag map (c), lam {lam} (scripts/why1_v3.py)", model=M, split="v3-test")
    (out / "summary.json").write_text(json.dumps(s, indent=1))
print("V3 DONE", flush=True)
