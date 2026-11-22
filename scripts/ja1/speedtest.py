"""ja1 speed test: replicas (stock) vs batched-cotangent vmap backward (is_grads_batched) on one prompt; same rows?
    python speedtest.py [model] [rlens|none] [B...]"""
import json
import sys
import time

import os  # noqa: E402
os.environ["JA1_FLA"] = "1" if "fla" in sys.argv else "0"
sys.path.insert(0, "/root/ja1/code")
import torch as t  # noqa: E402

from ja1_fit import LAYERS, load_model, prompt_records  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens.lrp import apply_lrp_rules_for_mode  # noqa: E402

M = sys.argv[1]
lrp = sys.argv[2]
Bs = [int(x) for x in sys.argv[3:] if x.isdigit()]
model = load_model(M)
r = prompt_records(model.tokenizer, "topic", 1)[0]
layers = LAYERS[M]
T = len(r["ids"]) - 1
d = model.d_model


def fwd(B):
    with apply_lrp_rules_for_mode(model, lrp):
        ids = t.tensor(r["ids"], device=model.input_device)[None].expand(B, -1)
        with ActivationRecorder(model.layers, at=[*layers, len(model.layers) - 1], start_graph_at=min(layers)) as rec, \
                t.enable_grad():
            model.forward(ids)
    return rec.activations[len(model.layers) - 1], [rec.activations[l] for l in layers]


def rows_rep(B, nrows):
    tgt, src = fwd(B)
    out = []
    for ds in range(0, nrows, B):
        cot = t.zeros_like(tgt)
        cot[t.arange(B), T, ds + t.arange(B)] = 1
        g = t.autograd.grad(tgt, src, cot, retain_graph=ds + B < nrows)
        out.append(g[3][:, r["pos"]].float().sum(1))
    return t.cat(out)


def rows_vmap(B, nrows):
    tgt, src = fwd(1)
    out = []
    for ds in range(0, nrows, B):
        cot = t.zeros((B, *tgt.shape), device=tgt.device, dtype=tgt.dtype)
        cot[t.arange(B), 0, T, ds + t.arange(B)] = 1
        g = t.autograd.grad(tgt, src, cot, retain_graph=ds + B < nrows, is_grads_batched=True)
        out.append(g[3][:, 0, r["pos"]].float().sum(1))
    return t.cat(out)


res = {}
NR = 256
a = rows_rep(4, 8)
for B in Bs:
    t.cuda.synchronize(); t0 = time.time()
    b = rows_vmap(B, NR)
    t.cuda.synchronize()
    print("vmap", B, "s/row", (time.time() - t0) / NR, "relF(first 8 rows vs rep4)", float((a - b[:8]).norm() / a.norm()),
          "peak", t.cuda.max_memory_allocated() / 1e9, flush=True)
t.cuda.synchronize(); t0 = time.time()
rows_rep(4, 64)
t.cuda.synchronize()
print("rep 4 s/row", (time.time() - t0) / 64, flush=True)
sys.exit()
for B in Bs:
    for name, f in (("rep", rows_rep), ("vmap", rows_vmap)):
        try:
            t.cuda.synchronize(); t.cuda.reset_peak_memory_stats(); t0 = time.time()
            R_ = f(B, 4 * B)
            t.cuda.synchronize()
            res[f"{name}{B}"] = dict(s_per_pass=(time.time() - t0) / 4, peak_gb=t.cuda.max_memory_allocated() / 1e9)
            res[f"{name}{B}"]["rows"] = R_
        except Exception as e:  # noqa: BLE001
            res[f"{name}{B}"] = dict(err=repr(e)[:300])
        print(name, B, {k: v for k, v in res[f"{name}{B}"].items() if k != "rows"}, flush=True)
    if "rows" in res.get(f"rep{B}", {}) and "rows" in res.get(f"vmap{B}", {}):
        a, b = res[f"rep{B}"]["rows"], res[f"vmap{B}"]["rows"]
        print("relF rep vs vmap", float((a - b).norm() / a.norm()), flush=True)
print(f"seq {len(r['ids'])} passes/prompt at B: {[d // B for B in Bs]}")
