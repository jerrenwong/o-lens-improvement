"""Fill the residual store with the fast subset's cells from the published dataset; optionally capture the rest.

Usage (pod, benchmark env; setup_pod.sh calls it):
  /root/wsb_env/bin/python pull_residuals.py --families fast|f1,f2 [--store DIR] [--report coverage.json]
         [--capture [--capture-dtype fp32|bf16] [--max-gpu-gib 74] [--max-cpu-gib 200]] [--check]
What it does, per family (wsbench.produce.store.cells_for with backend=None, as scripts/w5/pull_store.py):
  items already in the store under today's key are kept; the others are pulled from
  agu18dec/workspacebench_residuals when evals/residuals/map.json records the same key (only the part files
  that hold wanted rows are downloaded, into the pod-local HF cache, which is deleted after each family to keep
  the 50 GB disk free); items in neither place are listed in the report.
  --check: count only (no vector download; the dataset's index files are still fetched).
  --capture: forward-pass the missing items (README: 23 knowledge_cutoff fast items, 13,277 cells, are not
  republished) with Qwen3.6-27B. The published capture is fp32 (~108 GB of weights): on one 80 GB GPU it is
  loaded with device_map="auto" and the rest offloaded to CPU RAM (needs ~40 GB free RAM; slow but only ~23
  forward passes). --capture-dtype bf16 fits one GPU but is NOT the published numerics: it is written to the
  store's separate "bf16" folder and make_readouts.py uses it only with --allow-bf16.
  Weights are read from the local snapshot .hf27/hub/.../1cda14dd...: its 15 safetensors LFS oids equal those
  of the benchmark's DEFAULT_REVISION 6a9e13bd... (checked through the HF API tree on 2026-11-27).
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from wsbench import readplan
from wsbench.produce import dataset, store
from wsbench.produce.backend import DEFAULT_REVISION, Backend

R = "/workspace/o-lens-improvement"
# FIX 2026-11-27: the /workspace snapshot holds 0-byte placeholders (no symlinks on /workspace); use the pod-local
# cache built by scripts/fix_hf27_links.py (symlinks into the blobs on /workspace), or WSB_SNAP (e.g. a /dev/shm copy)
SNAP = os.environ.get("WSB_SNAP", "/root/hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5")

ap = argparse.ArgumentParser()
ap.add_argument("--families", default="fast")
ap.add_argument("--store", default=f"{R}/.hf27/wsb/store")
ap.add_argument("--report", default=f"{R}/runs/wsb/coverage_fast.json")
ap.add_argument("--check", action="store_true")
ap.add_argument("--capture", action="store_true")
ap.add_argument("--capture-dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--max-gpu-gib", type=int, default=74)
ap.add_argument("--max-cpu-gib", type=int, default=200)
ap.add_argument("--keep-hf-cache", action="store_true")
a = ap.parse_args()

if a.families == "fast":
    from wsbench import fast as fast_mod
    fams = sorted(fast_mod.subset())  # {family: entry} of the active version (fast-v6: 18 families)
else:
    fams = a.families.split(",")

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=DEFAULT_REVISION)
where = store.Store(a.store, DEFAULT_REVISION, "fp32")
pull = dataset.source("agu18dec/workspacebench_residuals", "", "fp32", DEFAULT_REVISION)
hf_ds = Path(os.environ.get("HF_HOME", "/root/hf_wsb")) / "hub" / "datasets--agu18dec--workspacebench_residuals"

rep = json.loads(Path(a.report).read_text()) if Path(a.report).exists() else {}
missing_all = {}
for fam in fams:
    t0 = time.time()
    specs = readplan.plan(fam, reads="full", fast=True)
    located = [store.locate(s, tok) for s in specs]
    stored = where.has(fam, {at.spec.id: at.key for at in located})
    in_ds = {at.spec.id for at in located if at.spec.id not in stored and pull.folder_of(at) is not None}
    miss = [s for s in specs if s.id not in stored and s.id not in in_ds]
    n_plan = sum(len(at.rows) for at in located)
    n_miss_cells = sum(len(at.rows) for at in located if at.spec.id in {s.id for s in miss})
    layers = sorted({int(la) for at in located for la, _, _ in at.rows})
    line = dict(items=len(specs), items_in_store=len(stored), items_in_dataset=len(in_ds),
                items_missing=[s.id for s in miss], cells_plan=n_plan, cells_missing=n_miss_cells,
                layers=layers)
    if not a.check:
        ok = [s for s in specs if s.id not in {m.id for m in miss}]
        cells, counts = store.cells_for(where, fam, ok, tok, pull=pull, backend=None)
        line.update(cells_now_in_store=len(cells), counts=counts.line(fam))
        if not a.keep_hf_cache and hf_ds.exists():
            shutil.rmtree(hf_ds / "blobs", ignore_errors=True)  # part files of this family; index files are tiny
    line["secs"] = round(time.time() - t0)
    rep[fam] = line
    if miss:
        missing_all[fam] = miss
    print(f"{fam}: items {len(specs)} = {len(stored)} store + {len(in_ds)} dataset + {len(miss)} missing; "
          f"cells {n_plan} ({n_miss_cells} missing); layers {layers}; {line['secs']}s", flush=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))

tot = {k: sum(v[k] for f, v in rep.items() if f in fams) for k in ("items", "cells_plan", "cells_missing")}
print(f"TOTAL {len(fams)} families: {tot['items']} items, {tot['cells_plan']} cells, {tot['cells_missing']} missing "
      f"({100 * tot['cells_missing'] / max(1, tot['cells_plan']):.1f}%)")

if a.capture and missing_all:
    import torch
    from transformers import AutoModelForCausalLM

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.capture_dtype]
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=dt, device_map="auto",
        max_memory={0: f"{a.max_gpu_gib}GiB", "cpu": f"{a.max_cpu_gib}GiB"})
    model.eval()
    print(f"model loaded ({a.capture_dtype}, device_map=auto) in {time.time() - t0:.0f}s; "
          f"devices {sorted({str(p.device) for p in model.parameters()})}", flush=True)
    be = Backend(model=model, tokenizer=tok, device="cuda", model_id="Qwen/Qwen3.6-27B")
    cap_store = store.Store(a.store, DEFAULT_REVISION, a.capture_dtype)
    for fam, miss in missing_all.items():
        t0 = time.time()
        cells, counts = store.cells_for(cap_store, fam, miss, tok, pull=None, backend=be)
        rep[fam]["captured"] = dict(dtype=a.capture_dtype, items=len(miss), cells=len(cells),
                                    secs=round(time.time() - t0))
        print(counts.line(fam), f"| captured in {time.time() - t0:.0f}s", flush=True)
        Path(a.report).write_text(json.dumps(rep, indent=1))
