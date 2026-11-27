"""Does our capture reproduce the published residuals? Re-capture N items that the dataset DOES hold and compare.

Usage (pod, benchmark env): CUDA_VISIBLE_DEVICES=1 /root/wsb_env/bin/python check_capture.py [--family knowledge_cutoff] [--n 3]
Loads Qwen3.6-27B exactly as pull_residuals.py --capture does (fp32, device_map=auto, CPU offload), forward-passes
N items already in the fp32 store (pulled from agu18dec/workspacebench_residuals), and prints per item the
relative L2 error and cosine of our vectors (rounded to bf16, as stored) against the published ones, per layer
(worst over positions). bf16 rounding alone gives a relative error of about 2e-3.
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsbench import readplan
from wsbench.produce import store
from wsbench.produce.backend import DEFAULT_REVISION, Backend

SNAP = os.environ.get("WSB_SNAP", "/root/hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5")
ap = argparse.ArgumentParser()
ap.add_argument("--family", default="knowledge_cutoff")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--store", default="/workspace/o-lens-improvement/.hf27/wsb/store")
ap.add_argument("--max-gpu-gib", type=int, default=72)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=DEFAULT_REVISION)
where = store.Store(a.store, DEFAULT_REVISION, "fp32")
specs = readplan.plan(a.family, reads="full", fast=True)
located = {s.id: store.locate(s, tok) for s in specs}
have, _ = where.find(a.family, {i: at.key for i, at in located.items()})
import json  # noqa: E402
from pathlib import Path  # noqa: E402
pulled = {e["id"] for p in Path(where.folder(a.family)).glob("*.json")
          if json.loads(p.read_text())["info"].get("source", "").startswith("pulled")
          for e in json.loads(p.read_text())["items"]}
pick = sorted((i for i in have if i in pulled), key=lambda i: len(have[i].rows))[: a.n]  # shortest PULLED items
print("comparing pulled items:", pick, flush=True)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32, device_map="auto",
                                             max_memory={0: f"{a.max_gpu_gib}GiB", "cpu": "300GiB"}).eval()
be = Backend(model=model, tokenizer=tok, device="cuda", model_id="Qwen/Qwen3.6-27B")
fresh = store.capture([located[i] for i in pick], be)
for it in fresh:
    ref = have[it.id].acts.float()
    ours = it.acts.to(torch.bfloat16).float()
    assert it.rows == have[it.id].rows
    by_l = {}
    for k, (la, _p, _t) in enumerate(it.rows):
        rel = ((ours[k] - ref[k]).norm() / ref[k].norm()).item()
        cos = torch.nn.functional.cosine_similarity(ours[k], ref[k], dim=0).item()
        w = by_l.setdefault(la, [0.0, 1.0])
        w[0], w[1] = max(w[0], rel), min(w[1], cos)
    print(f"{it.id}: {len(it.rows)} cells; worst rel L2 / min cos per layer:",
          " ".join(f"L{la}:{v[0]:.1e}/{v[1]:.5f}" for la, v in sorted(by_l.items())), flush=True)
