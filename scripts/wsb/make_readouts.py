"""WorkspaceBench token-list readouts for one lens, from the residual store (no model forward pass).

Usage (GPU pod, benchmark env with the .pth bridge from setup_pod.sh):
  /root/wsb_env/bin/python make_readouts.py --lens jlens|jpp|alens|jpp+alens|custom
        [--candidate auto/candidates/X.py] [--name DIR_NAME] [--families fast|f1,f2] [--layers grid|alens|20,44]
        [--filter f4|none] [--lam 0.5] [--limit N] [--out-root DIR] [--allow-bf16] [--validate N]
  e.g.  .../make_readouts.py --lens jpp --families poetry            -> .hf27/wsb/readouts/jpp/poetry.jsonl
Lenses (formulas as scripts/w5/make_readouts.py and auto/candidates/alens.py; all decode to the 27B vocabulary):
  jlens      public J-lens (artifacts/baselines/jlens27b_prefit.pt = neuronpedia n=1000): (W_U J_L h) / ||J_L^T W_U[t]||,
             the runner's own JLens readout (wsbench/produce/methods.py JLens.read); w5 validated identical lists
  jpp        released J++ (artifacts/baselines/jpp27b_released.pt): final_norm(unit(J++_L h)) @ W_U^T
  alens      a(h) = unit(unit(E_t3 h + b) + unit(E_s5 h + b'))  (.hf27/ask3/E_27b_t3.pt, .hf27/ask5/E_27b_s5.pt)
  jpp+alens  unit(J++ h) + lam * a(h), lam = 0.5 (the fit-chosen value; --lam to change)
  custom     any candidate file with build(ctx) -> lens with .transport(h, layer, resid_all) returning a
             final-layer-basis vector (decoded as final_norm(v) @ W_U^T, as the auto/ harness does). ctx is a
             MINIMAL context: model (d_model, tokenizer, _lm_head.weight), J, JPP (fp32 dicts at the chosen
             layers), device, readout_layers; ctx.fit / ctx.filters / ctx.wikitext are None, so a candidate that
             chooses a weight on the fit split must be given a fixed one (e.g. its env var) or it fails here.
             resid_all is passed as None (ASSUMPTION: the candidate does not need other layers' residuals).
Layers: default = the 11-layer grid 20..60 step 4 for jlens/jpp (both lens files hold layers 0..62) and
  24,32,40,48,56 for alens / jpp+alens (A-lens maps exist at 8..56 step 8). A file with fewer layers than the
  plan must be judged with layers=<those layers> (judge.sh reads them from the sidecar).
Filter f4 (auto/filters.py, the project's "full filter") adapted to the benchmark: masked = tokens with no letter
  or digit, special/added tokens, tokens containing "@", "&", "<" or ">"; top-10 = best member of each of the
  top 10 "count each word once" groups (normal_form). The f1 base and the "spared" ids of auto/filters.py are
  our own bench's answer lists; with no bench they are empty here (ASSUMPTION: fine for WSB, whose answers are
  words/numbers, never punctuation). --filter none = plain top-10 (the runner's J-lens arm; use it with --lens
  jlens to reproduce the published `jlens-public` readouts).
Output: <out-root>/<name>/<family>.jsonl, one row per (item, layer, position) of the family's FAST plan rows,
  {"id","layer","pos","token","tokens","scores"}; then the sidecar <family>.jsonl.meta.json (written last:
  {reads: full, sampling: tokens10, fast: fast-v6, plan_sha: wsbench.reads.fast_plan_sha(family), layers, ...}).
  Items whose residuals are not in the store are skipped and listed in <out-root>/<name>/_make_readouts_summary.json
  (judge them with allow_missing=True, the row is then incomplete).
"""
import argparse
import importlib.util
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

from wsbench import fast as fast_mod  # noqa: E402
from wsbench import readplan  # noqa: E402
from wsbench import reads as reads_mod  # noqa: E402
from wsbench.produce import store  # noqa: E402
from wsbench.produce.backend import DEFAULT_REVISION  # noqa: E402
from wsbench.produce.render import display_tokens  # noqa: E402
from wsbench.readouts import sidecar_path, write_sidecar  # noqa: E402

R = "/workspace/o-lens-improvement"
# FIX 2026-11-27: the /workspace snapshot holds 0-byte placeholders (no symlinks on /workspace); use the pod-local
# cache built by scripts/fix_hf27_links.py (symlinks into the blobs on /workspace), or WSB_SNAP (e.g. a /dev/shm copy)
SNAP = os.environ.get("WSB_SNAP", "/root/hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5")
GRID = list(range(20, 61, 4))
ALENS_LAYERS = [24, 32, 40, 48, 56]
K = 10
BATCH = 512

ap = argparse.ArgumentParser()
ap.add_argument("--lens", required=True, choices=["jlens", "jpp", "alens", "jpp+alens", "custom"])
ap.add_argument("--candidate", default="")
ap.add_argument("--name", default="")
ap.add_argument("--families", default="fast")
ap.add_argument("--layers", default="")
ap.add_argument("--filter", default="f4", choices=["f4", "none"])
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0, help="first N fast items per family (smoke test only)")
ap.add_argument("--out-root", default=f"{R}/.hf27/wsb/readouts")
ap.add_argument("--store", default=f"{R}/.hf27/wsb/store")
ap.add_argument("--allow-bf16", action="store_true", help="also read items captured in bf16 (not the published numerics)")
ap.add_argument("--validate", type=int, default=0, help="jlens --filter none: compare N cells/family with wsbench's JLens")
a = ap.parse_args()

if a.lens == "custom" and not a.candidate:
    sys.exit("--lens custom needs --candidate <file>")
if a.layers in ("", "default"):
    LAYERS = ALENS_LAYERS if a.lens in ("alens", "jpp+alens") else (GRID if a.lens != "custom" else None)
    if LAYERS is None:
        sys.exit("--lens custom needs --layers (grid | alens | csv)")
elif a.layers == "grid":
    LAYERS = GRID
elif a.layers == "alens":
    LAYERS = ALENS_LAYERS
else:
    LAYERS = sorted(int(x) for x in a.layers.split(","))
if a.lens in ("alens", "jpp+alens") and not set(LAYERS) <= set(range(8, 57, 8)):
    sys.exit(f"A-lens maps exist at 8..56 step 8 only; asked {LAYERS}")
name = a.name or (a.lens.replace("+", "_") if a.lens != "custom" else os.path.splitext(os.path.basename(a.candidate))[0])
out_dir = f"{a.out_root}/{name}"
os.makedirs(out_dir, exist_ok=True)
fams = sorted(fast_mod.subset()) if a.families == "fast" else a.families.split(",")
dev = "cuda"
torch.set_grad_enabled(False)


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def tensor(nm, file):
    with safe_open(f"{SNAP}/{file}", "pt") as f:
        return f.get_tensor(nm)


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=DEFAULT_REVISION)
W_U_bf = tensor("lm_head.weight", "model-00008-of-00015.safetensors").to(dev)  # bf16, as Backend.load
W_U = W_U_bf.float()
V, D = W_U.shape
norm = Qwen3_5RMSNorm(D, eps=1e-6).to(dev)
norm.weight.data = tensor("model.language_model.norm.weight", "model-00015-of-00015.safetensors").to(dev).float()


def load_J():
    J = torch.load(f"{R}/artifacts/baselines/jlens27b_prefit.pt", map_location="cpu", weights_only=False)["J"]
    return {l: (J[l] if l in J else J[str(l)]).to(dev).float() for l in LAYERS}


def load_JPP():
    J = load_lens_file(f"{R}/artifacts/baselines/jpp27b_released.pt", hf_model_name="Qwen/Qwen3.6-27B").jacobians_L_dict_FN
    J = {int(l): v for l, v in J.items()}
    return {l: J[l].to(dev).float() for l in LAYERS}


def load_maps(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items() if l in LAYERS}


def emap(E, h, l):
    mx, W, my = E[l]
    return (h - mx) @ W + my


# ---- the lens: scores(H [n, D] fp32, layer) -> [n, V]
if a.lens == "jlens":
    Jp = load_J()
    denom = {l: (W_U @ Jp[l]).norm(dim=1).clamp_min(1e-9) for l in LAYERS}

    def scores(H, l):
        return ((H @ Jp[l].T) @ W_U.T) / denom[l]
elif a.lens == "jpp":
    JPP = load_JPP()

    def scores(H, l):
        return norm(unit(H @ JPP[l].T)) @ W_U.T
elif a.lens in ("alens", "jpp+alens"):
    E1 = load_maps(f"{R}/.hf27/ask3/E_27b_t3.pt")
    E2 = load_maps(f"{R}/.hf27/ask5/E_27b_s5.pt")
    assert all(l in E1 and l in E2 for l in LAYERS), "A-lens maps missing a layer"
    JPP = load_JPP() if a.lens == "jpp+alens" else None

    def scores(H, l):
        v = unit(unit(emap(E1, H, l)) + unit(emap(E2, H, l)))
        if JPP is not None:
            v = unit(H @ JPP[l].T) + a.lam * v
        return norm(v) @ W_U.T
else:
    spec = importlib.util.spec_from_file_location("wsb_candidate", a.candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.candidate)))
    spec.loader.exec_module(mod)
    ctx = SimpleNamespace(
        model=SimpleNamespace(d_model=D, tokenizer=tok, _lm_head=SimpleNamespace(weight=W_U_bf)),
        J=load_J(), JPP=load_JPP(), device=dev, readout_layers=list(LAYERS),
        fit=None, filters=None, dict_path=None, wikitext=None)
    lens = mod.build(ctx)

    def scores(H, l):
        v = lens.transport(H.clone(), l, None)
        if not isinstance(v, torch.Tensor) or tuple(v.shape) != tuple(H.shape):
            raise ValueError(f"transport at layer {l} returned {getattr(v, 'shape', type(v))}, expected {tuple(H.shape)}")
        return norm(v.float()) @ W_U.T

# ---- filter f4 (auto/filters.py, no bench lists)
EXCL = GROUP = None
if a.filter == "f4":
    _fs = importlib.util.spec_from_file_location("w1_filters", f"{R}/auto/filters.py")
    _fm = importlib.util.module_from_spec(_fs)
    _fs.loader.exec_module(_fm)
    F = _fm.Filters(tok, V, []).get("f4")
    EXCL = F.excluded_ids.to(dev)
    GROUP = F.group_ids.tolist()
    print(f"filter f4: {len(EXCL)} of {V} ids masked, {F.n_groups} groups; parts {F.parts}", flush=True)


def topk_rows(S):
    """[n, V] -> (ids [n][K], vals [n][K]) after the filter."""
    if EXCL is None:
        v, ids = S.topk(K, -1)
        return ids.tolist(), v.tolist()
    S[:, EXCL] = float("-inf")
    v, ids = S.topk(256, -1)
    v, ids = v.tolist(), ids.tolist()
    out_i, out_v = [], []
    for r in range(len(ids)):
        seen, oi, ov = set(), [], []
        for t, x in zip(ids[r], v[r]):
            g = GROUP[t]
            if g not in seen:
                seen.add(g)
                oi.append(t)
                ov.append(x)
                if len(oi) == K:
                    break
        if len(oi) < K:  # rare: fewer than K groups in the top 256, use the exact helper on the full row
            oi = _fm.group_topk(S[r], F.group_ids, K)
            ov = S[r, oi].tolist()
        out_i.append(oi)
        out_v.append(ov)
    return out_i, out_v


ref = None
if a.validate and a.lens == "jlens" and a.filter == "none":
    import huggingface_hub
    from wsbench.produce import methods
    huggingface_hub.hf_hub_download = lambda repo, filename, **kw: f"{R}/artifacts/baselines/jlens27b_prefit.pt"
    ref = methods.JLens()
    ref.bind(SimpleNamespace(device=dev, unembed=W_U_bf, tokenizer=tok))

wheres = [store.Store(a.store, DEFAULT_REVISION, "fp32")]
if a.allow_bf16:
    wheres.append(store.Store(a.store, DEFAULT_REVISION, "bf16"))

sum_path = f"{out_dir}/_make_readouts_summary.json"
T_ALL, N_ALL = time.time(), 0
torch.cuda.reset_peak_memory_stats()
summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
for fam in fams:
    t0 = time.time()
    specs = readplan.plan(fam, reads="full", fast=True)
    if a.limit:
        specs = specs[: a.limit]
    located = {s.id: store.locate(s, tok) for s in specs}
    cells, src = [], {}
    left = list(specs)
    for w in wheres:
        have = w.has(fam, {i: at.key for i, at in located.items()})
        got = [s for s in left if s.id in have]
        if got:
            cs, _ = store.cells_for(w, fam, got, tok, pull=None, backend=None)
            cells += cs
            src[w.dtype] = len(got)
        left = [s for s in left if s.id not in have]
    order = {s.id: k for k, s in enumerate(specs)}
    cells = [c for c in cells if c.layer in LAYERS]
    cells.sort(key=lambda c: order[c.id])  # plan order; within an item the store's own order is kept (stable sort)
    rows = [None] * len(cells)
    n_val = n_eq = 0
    for l in LAYERS:
        idx = [i for i, c in enumerate(cells) if c.layer == l]
        for b0 in range(0, len(idx), BATCH):
            bi = idx[b0:b0 + BATCH]
            H = torch.stack([cells[i].h for i in bi]).to(dev).float()
            ids, vals = topk_rows(scores(H, l))
            for j, i in enumerate(bi):
                c = cells[i]
                rows[i] = {"id": c.id, "layer": c.layer, "pos": c.pos, "token": c.token,
                           "tokens": display_tokens(tok, ids[j]), "scores": [round(x, 4) for x in vals[j]]}
        if ref is not None and idx:  # validation spread over every layer and the whole family (seeded sample)
            import random
            per_l = -(-a.validate // len(LAYERS))
            for i in random.Random(f"{fam}{l}").sample(idx, min(per_l, len(idx))):
                n_val += 1
                same = ref.read(cells[i].h.float(), l).tokens == rows[i]["tokens"]
                n_eq += same
                if not same and n_val - n_eq <= 3:
                    print("  MISMATCH", fam, cells[i].id, l, cells[i].pos, ref.read(cells[i].h.float(), l).tokens,
                          rows[i]["tokens"], flush=True)
    p = f"{out_dir}/{fam}.jsonl"
    sc = sidecar_path(p)
    if os.path.exists(sc):
        os.remove(sc)  # never leave a current-looking sidecar beside a file being rewritten
    with open(p, "w", encoding="utf-8") as f:  # written in place (/workspace mishandles renames over files)
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_sidecar(p, reads="full", sampling="tokens10", reader=f"wsb-{a.lens}-{a.filter}", arm=name,
                  layers=LAYERS, fast=fast_mod.VERSION, plan_sha=reads_mod.fast_plan_sha(fam),
                  lens=a.lens, filter=a.filter, lam=a.lam if a.lens == "jpp+alens" else None,
                  candidate=a.candidate or None, limit=a.limit or None)
    summary[fam] = dict(items=len(specs), items_missing=[s.id for s in left], items_by_store_dtype=src,
                        cells=len(cells), layers=LAYERS, filter=a.filter, secs=round(time.time() - t0),
                        **({"validated": n_val, "identical_lists": n_eq} if ref is not None else {}))
    print(fam, {k: (len(v) if k == "items_missing" else v) for k, v in summary[fam].items()}, flush=True)
    json.dump(summary, open(sum_path, "w"), indent=1)
    N_ALL += len(cells)
dt = time.time() - T_ALL
print(f"DONE {name}: {N_ALL} cells in {dt:.0f}s = {N_ALL / max(dt, 1e-9):.0f} cells/s; peak GPU memory "
      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved",
      flush=True)
