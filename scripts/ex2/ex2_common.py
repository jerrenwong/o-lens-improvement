"""ex2 shared config and loaders. All inputs read-only; outputs under runs/ex2/.

Fits (stamps checked 2026-11-22):
  9B, one router (runs/r2_9b/A/router_K8.pt, raw WikiText, skip 16) shared by all 9B fits, so expert e is the
  same cluster in every 9B fit:
    9b_64a  experts from shards 0-1 (48 prompts), weights fitted on the 75 FIT items
    9b_64b  experts from shards 2-3 (48 other prompts), same FIT items
    9b_full experts from shards 0-3 (96 prompts), same FIT items
    9b_w2   a second fit-weights run on the 64a experts (same args; weights turn out bit-identical)
  2B, four separately fitted routers + experts (128 prompts each), weights on the 52 FIT items:
    2b_A raw WikiText, skip 16 (the recipe) | 2b_B natural format, skip 16 | 2b_C natural, skip 1 | 2b_D raw, skip 1
    (expert indices are NOT aligned across 2B fits: different routers)
"""
import json
import os

R = "/workspace/o-lens-improvement"
OUT = f"{R}/runs/ex2"
os.makedirs(OUT, exist_ok=True)

MODEL = {"9b": "Qwen/Qwen3.5-9B", "2b": "Qwen/Qwen3.5-2B"}
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}
BENCH = {"9b": f"{R}/artifacts/bench/qwen3.5-9b.pt", "2b": f"{R}/artifacts/bench/qwen3.5-2b.pt"}

_9 = f"{R}/runs/r2_9b/A"
_2 = f"{R}/runs/r1_2b"
FITS = {
    "9b_64a": dict(m="9b", experts=f"{_9}/A_64a/experts_K8.pt", weights=f"{_9}/A_64a/jpp_A_64a_expert_weights.json",
                   L=f"{_9}/A_64a/jpp_A_64a.pt", pooled=f"{_9}/A_64a/pooled_A_64a.pt"),
    "9b_64b": dict(m="9b", experts=f"{_9}/A_64b/experts_K8.pt", weights=f"{_9}/A_64b/jpp_A_64b_expert_weights.json",
                   L=f"{_9}/A_64b/jpp_A_64b.pt", pooled=f"{_9}/A_64b/pooled_A_64b.pt"),
    "9b_full": dict(m="9b", experts=f"{_9}/A_full/experts_K8.pt", weights=f"{_9}/A_full/jpp_A_full_expert_weights.json",
                    L=f"{_9}/A_full/jpp_A_full.pt", pooled=f"{_9}/A_full/pooled_A_full.pt"),
    "9b_w2": dict(m="9b", experts=f"{_9}/A_64a/experts_K8.pt", weights=f"{_9}/A_64a_w2/jpp_A_64a_w2_expert_weights.json",
                  L=f"{_9}/A_64a_w2/jpp_A_64a_w2.pt", pooled=f"{_9}/A_64a/pooled_A_64a.pt"),
}
for c in "ABCD":
    FITS[f"2b_{c}"] = dict(m="2b", experts=f"{_2}/{c}/experts_K8.pt", weights=f"{_2}/{c}/jpp_{c}_expert_weights.json",
                           L=f"{_2}/{c}/jpp_{c}.pt", pooled=f"{_2}/{c}/pooled_{c}.pt")
MAIN = {"9b": "9b_64a", "2b": "2b_A"}

RELEASED = {
    "9b": dict(jpp=f"{R}/artifacts/baselines/jpp9b_prefit.pt", jl=f"{R}/runs/v6_9b/jlens9b.pt"),
    "2b": dict(jpp=f"{R}/artifacts/baselines/jpp_prefit.pt", jl=f"{R}/artifacts/baselines/jlens_prefit.pt"),
    "27b": dict(jpp=f"{R}/artifacts/baselines/jpp27b_released.pt", jl=f"{R}/artifacts/baselines/jlens27b_prefit.pt"),
}
LAYERS["27b"] = [8, 16, 24, 32, 40, 48, 56]


def load_weights(fit):
    W = json.load(open(FITS[fit]["weights"]))["by_layer"]
    return {int(k): v["weights_unit_l2"] for k, v in W.items()}


def load_matrix_dict(path, layers=None):
    """{layer: fp32 cpu tensor} from either file layout (jpp_lens lens file or {'J': {...}} prefit dict)."""
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "J" in d:
        M = d["J"]
    else:
        M = d["parameters"]["jacobians"]
    M = {int(k): v for k, v in M.items()}
    if layers is not None:
        M = {k: M[k] for k in layers}
    return {k: v.float() for k, v in M.items()}


def load_experts(fit):
    """(experts {layer: [E,F,N] fp16 cpu}, counts {layer: [E]}, pooled_fp16 {layer})"""
    import torch
    s = torch.load(FITS[fit]["experts"], map_location="cpu", weights_only=True, mmap=True)
    return ({int(k): v for k, v in s["experts"].items()}, {int(k): v.double() for k, v in s["position_counts"].items()},
            {int(k): v for k, v in s["pooled"].items()}, {int(k): v for k, v in s["fallback"].items()})


def content_experts():
    """Label-free 'content' experts from ex1's cluster token statistics (WikiText windows, no labels):
    cluster whose positions are >= 50% content-word starts + word continuations (subword pieces).
    Only for the routers ex1 analysed: 9B router A (all 9B fits) and 2B fit A."""
    out = {}
    for tag, fits in (("9b", ["9b_64a", "9b_64b", "9b_full", "9b_w2"]), ("2b", ["2b_A"])):
        S = json.load(open(f"{R}/runs/ex1/{tag}/summary_{tag}.json"))
        sel = {}
        for L, d in S["layers"].items():
            sel[int(L)] = [ex["e"] for ex in d["experts"]
                           if ex["kinds"]["content_start"] + ex["kinds"]["continuation"] >= 0.5]
        for f in fits:
            out[f] = sel
    return out


def null_weights(w, n=2000, seed=0):
    """Null weight vectors for a fitted 8-vector w (numpy, float64):
    sign : all 256 sign patterns of |w| (exact enumeration; includes w itself and -w)
    perm : n random permutations of w across the 8 experts
    dir  : n random non-negative mixtures, Dirichlet(1,...,1) (unit L1)
    gauss: n isotropic Gaussian directions (unit L2)"""
    import itertools

    import numpy as np
    w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(seed)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(w))))
    perm = np.stack([w[rng.permutation(len(w))] for _ in range(n)])
    dirich = rng.dirichlet(np.ones(len(w)), size=n)
    g = rng.standard_normal((n, len(w)))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return dict(sign=signs * np.abs(w)[None], perm=perm, dir=dirich, gauss=g)
