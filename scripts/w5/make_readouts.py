"""W5: token-lens readouts for WorkspaceBench (Qwen3.6-27B) from the stored residuals, no model load.

Arms (top 10 tokens + scores per cell, all with NO token filter, as the runner's J-lens adapter):
  jlens_pub      the runner's J-lens formula: (W_U J_L h) / ||J_L^T W_U[t]||   (neuronpedia n=1000 J, == jlens27b_prefit.pt)
  jpp            unit(J++ h)                         -> final norm + W_U  (our decoding)
  alens          a(h) = unit(unit(E_t3 h + b) + unit(E_s5 h + b'))  -> final norm + W_U
  jpp_a<lam>     unit(J++ h) + lam * a(h)            -> final norm + W_U   (lam sweep; lam chosen on our fit prompts = 0.5)
A-lens maps exist at 8..56 step 8, so the readouts cover layers 24, 32, 40, 48, 56 only (LAYERS env to change).
Writes <out>/<arm>/<family>.jsonl + sidecar (reads full, sampling tokens10, plan_sha of the family's plan).
Validation: VALIDATE=n compares jlens_pub with the runner's own methods.JLens.read on n cells per family.

Run (pod, wsbench venv; torch etc. come from /root/jpp_lens via a .pth):
  HF_HOME=/workspace/o-lens-improvement/.hf27 HF_HUB_OFFLINE=1 /root/wsbench/.venv/bin/python make_readouts.py <out> fam1,fam2 [tier=full|fast]
"""
import json
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import torch
from safetensors import safe_open

sys.path.append("/root/jpp_lens/src")
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

from wsbench import reads as reads_mod  # noqa: E402
from wsbench import readplan  # noqa: E402
from wsbench.produce import methods, store  # noqa: E402
from wsbench.produce.backend import DEFAULT_REVISION  # noqa: E402
from wsbench.produce.render import display_tokens  # noqa: E402
from wsbench.readouts import write_sidecar  # noqa: E402

R = "/workspace/o-lens-improvement"
STORE = f"{R}/.hf27/w5/residuals"
SNAP = f"{R}/.hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5"  # same lm_head/norm/tokenizer blobs as {DEFAULT_REVISION}
LAYERS = [int(x) for x in os.environ.get("LAYERS", "24,32,40,48,56").split(",")]
LAMS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
VALIDATE = int(os.environ.get("VALIDATE", "0"))
K = 10
dev = "cuda"
torch.set_grad_enabled(False)

out_root, fams = sys.argv[1], sys.argv[2].split(",")
fast = len(sys.argv) > 3 and sys.argv[3] == "fast"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def tensor(name, file):
    with safe_open(f"{SNAP}/{file}", "pt") as f:
        return f.get_tensor(name)


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=DEFAULT_REVISION,
                                    cache_dir=f"{R}/.hf27/w5/hf/hub")
W_U_bf = tensor("lm_head.weight", "model-00008-of-00015.safetensors").to(dev)  # bf16, as Backend.load
W_U = W_U_bf.float()
norm = Qwen3_5RMSNorm(W_U.shape[1], eps=1e-6).to(dev)
norm.weight.data = tensor("model.language_model.norm.weight", "model-00015-of-00015.safetensors").to(dev).float()

Jpub = torch.load(f"{R}/artifacts/baselines/jlens27b_prefit.pt", map_location="cpu", weights_only=False)["J"]
Jpub = {l: (Jpub[l] if l in Jpub else Jpub[str(l)]).to(dev).float() for l in LAYERS}
denom = {l: (W_U @ Jpub[l]).norm(dim=1).clamp_min(1e-9) for l in LAYERS}
JPP = load_lens_file(f"{R}/artifacts/baselines/jpp27b_released.pt", hf_model_name="Qwen/Qwen3.6-27B").jacobians_L_dict_FN
JPP = {int(l): v for l, v in JPP.items()}
JPP = {l: JPP[l].to(dev).float() for l in LAYERS}


def load_maps(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items() if l in LAYERS}


E1 = load_maps(f"{R}/.hf27/ask3/E_27b_t3.pt")
E2 = load_maps(f"{R}/.hf27/ask5/E_27b_s5.pt")
assert all(l in E1 and l in E2 for l in LAYERS), "A-lens maps missing a layer"


def emap(E, h, l):
    mx, W, my = E[l]
    return (h - mx) @ W + my


def arms(H, l):
    """H [n, d] fp32 -> {arm: logits-like scores [n, V]} computed lazily (generator)."""
    yield "jlens_pub", ((H @ Jpub[l].T) @ W_U.T) / denom[l]
    j = unit(H @ JPP[l].T)
    a = unit(unit(emap(E1, H, l)) + unit(emap(E2, H, l)))
    yield "jpp", norm(j) @ W_U.T
    yield "alens", norm(a) @ W_U.T
    for lam in LAMS:
        yield f"jpp_a{lam:g}", norm(j + lam * a) @ W_U.T


ARMS = ["jlens_pub", "jpp", "alens"] + [f"jpp_a{lam:g}" for lam in LAMS]
VOCAB = None

# runner's own J-lens for validation (bound to a minimal backend: bf16 lm_head, as Backend.load)
ref = None
if VALIDATE:
    ref = methods.JLens()
    # the hub-cache symlink of the neuronpedia file reads as empty on the network FS; hand the runner's own
    # loader the byte copy of that file (artifacts/baselines/jlens27b_prefit.pt, `cat` of the snapshot, m27 chain1)
    import huggingface_hub
    huggingface_hub.hf_hub_download = lambda repo, filename, **kw: f"{R}/artifacts/baselines/jlens27b_prefit.pt"
    ref.bind(SimpleNamespace(device=dev, unembed=W_U_bf, tokenizer=tok))

summary = {}
for fam in fams:
    t0 = time.time()
    specs = readplan.plan(fam, reads="full", fast=fast)
    where = store.Store(STORE, DEFAULT_REVISION, "fp32")
    located = [store.locate(s, tok) for s in specs]
    have = where.has(fam, {at.spec.id: at.key for at in located})
    specs = [replace(s, layers=list(LAYERS)) for s in specs if s.id in have]
    n_missing = len(located) - len(specs)
    # cells_for checks rows against the plan; ask for the full-grid specs, keep our layers
    full_specs = [s for s in readplan.plan(fam, reads="full", fast=fast) if s.id in have]
    cells, counts = store.cells_for(where, fam, full_specs, tok, pull=None, backend=None)
    cells = [c for c in cells if c.layer in LAYERS]
    fhs = {}
    for arm in ARMS:
        os.makedirs(f"{out_root}/{arm}", exist_ok=True)
        p = f"{out_root}/{arm}/{fam}.jsonl"
        if os.path.exists(p + ".meta.json"):
            os.remove(p + ".meta.json")
        fhs[arm] = open(p, "w", encoding="utf-8")
    n_val = n_eq = n_set = 0
    max_sdiff = 0.0
    # group by layer, in plan order within each layer group; output order = plan order
    rows = {arm: [None] * len(cells) for arm in ARMS}
    for l in LAYERS:
        idx = [i for i, c in enumerate(cells) if c.layer == l]
        for b0 in range(0, len(idx), 512):
            bi = idx[b0:b0 + 512]
            H = torch.stack([cells[i].h for i in bi]).to(dev).float()
            for arm, S in arms(H, l):
                v, ids = S.topk(K, -1)
                v, ids = v.tolist(), ids.tolist()
                for j, i in enumerate(bi):
                    c = cells[i]
                    rows[arm][i] = {"id": c.id, "layer": c.layer, "pos": c.pos, "token": c.token,
                                    "tokens": display_tokens(tok, ids[j]), "scores": [round(x, 4) for x in v[j]]}
            if ref is not None and n_val < VALIDATE:
                for i in bi[:max(0, VALIDATE - n_val)][:: max(1, len(bi) // 64)]:
                    rr = ref.read(cells[i].h.float(), l)
                    mine = rows["jlens_pub"][i]
                    n_val += 1
                    n_eq += rr.tokens == mine["tokens"]
                    n_set += set(rr.tokens) == set(mine["tokens"])
                    max_sdiff = max(max_sdiff, max(abs(a - b) for a, b in zip(rr.scores, mine["scores"])))
    for arm in ARMS:
        fhs[arm].write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[arm]))
        fhs[arm].close()
        write_sidecar(f"{out_root}/{arm}/{fam}.jsonl", reads="full", sampling="tokens10", reader=f"w5-{arm}",
                      arm=arm, layers=LAYERS,
                      plan_sha=reads_mod.fast_plan_sha(fam) if fast else reads_mod.plan_sha(fam))
    summary[fam] = dict(items=len(specs), items_missing=n_missing, cells=len(cells), validated=n_val,
                        identical_lists=n_eq, same_sets=n_set, max_score_diff=max_sdiff, secs=round(time.time() - t0))
    print(fam, summary[fam], flush=True)
p = f"{out_root}/_make_readouts_summary.json"
old = json.load(open(p)) if os.path.exists(p) else {}
old.update(summary)
json.dump(old, open(p, "w"), indent=1)
