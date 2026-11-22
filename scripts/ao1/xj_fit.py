"""ao1 XJ variants (orchestrator request): extra experts inside J++'s own weight fit. GPU, lite model (final norm +
LM head), FIT rows of the bench only; nothing here touches DEV.

Per layer, a basis of transports T_k(h) (k = sources below) is weighted by the authors' search, imported unchanged
from /root/jpp_lens (fit_expert_weights + build_search_starting_points + CondenseConfig defaults, exact objective
= that layer's pair macro pass@10 on the FIT items, signed unit-L2 weights, excluded token ids masked), on the
cached per-source logits (expert_logits), exactly as jpp_lens fit-weights does for its 8 experts.
Sources (--src, repeatable, order = weight order):
  exp:PATH        merged experts file (jpp_lens merge-experts output, key 'experts' {layer: [E, F, N]}) -> E terms
  mat:PATH        one matrix per layer (plain dict {layer: [F, N]}, or a jpp_lens lens file)       -> 1 term
  map:PATH        regression ask-map, affine: T = (h - mx) W + my (bias b kept, scaled by the same weight)
  perm:exp:PATH / perm:mat:PATH   control: the same matrices with the layer assignment shifted cyclically
                  (layer l uses the matrices fitted for the next readout layer; the last uses the first)
Output: runs/ao1/<tag>/xj/<name>.pt = {layer: {'M': [F, N], 'c': [F]}} with v = h @ M^T + c, and <name>.json
(weights, per-layer train macro, FIT pass@10 / pass@1 of the combined lens with the scorer's own metric).
    python xj_fit.py --model 9b --name J1 --src exp:A.pt --src mat:A2.pt
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/perf1")
sys.path.insert(0, "/root/jpp_lens/src")
from lite_model import get_lite_model  # noqa: E402
from olens.readout import Bench, readout_rows  # noqa: E402
from workspace_lens.fitting.condense_experts.fitter import (  # noqa: E402
    CondenseConfig, build_search_starting_points, fit_expert_weights)
from workspace_lens.fitting.condense_experts.rank_objective import (  # noqa: E402
    build_pair_table, expert_logits, expert_pair_pass_scores)

MODEL = {"9b": "Qwen/Qwen3.5-9B", "2b": "Qwen/Qwen3.5-2B"}
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}
BENCH = {"9b": f"{R}/artifacts/bench/qwen3.5-9b.pt", "2b": f"{R}/artifacts/bench/qwen3.5-2b.pt"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["9b", "2b"], required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--src", action="append", required=True)
args = ap.parse_args()
m, Ls = args.model, LAYERS[args.model]
torch.set_grad_enabled(False)
dev = "cuda"
CFG = CondenseConfig()
spec = importlib.util.spec_from_file_location("ao1_ev", f"{R}/auto/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def load_mat(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "parameters" in d:
        d = d["parameters"]["jacobians"]
    elif isinstance(d, dict) and "J" in d:
        d = d["J"]
    if not all(isinstance(k, int) or str(k).isdigit() for k in d):
        from workspace_lens.utils import load_lens_file
        d = load_lens_file(path, hf_model_name=MODEL[m]).jacobians_L_dict_FN
    return {int(k): v.float()[None] for k, v in d.items()}  # [1, F, N]


def load_exp(path):
    s = torch.load(path, map_location="cpu", weights_only=True)
    return {int(k): v.float() for k, v in s["experts"].items()}


def load_src(s):
    perm = s.startswith("perm:")
    s = s[5:] if perm else s
    kind, path = s.split(":", 1)
    if kind == "map":
        mp = torch.load(path, map_location="cpu")
        return "map", {int(l): (v["mx"].float(), v["W"].float() * v["scale"], v["my"].float()) for l, v in mp.items()}
    M = load_exp(path) if kind == "exp" else load_mat(path)
    if perm:
        M = {l: M[Ls[(i + 1) % len(Ls)]] for i, l in enumerate(Ls)}
    return "mat", M


srcs = [(s, *load_src(s)) for s in args.src]
bench = Bench.load(BENCH[m])
fi = bench.index("fit")
model = get_lite_model(MODEL[m])
V = model._lm_head.weight.shape[0]
excl = torch.zeros(V, dtype=torch.bool, device=dev)
excl[bench.excluded_ids.to(dev)] = True
slugs = [bench.eval_slugs[i] for i in fi]
cands = [bench.candidates[i] for i in fi]
pairs = build_pair_table(cands, slugs, list(range(len(fi))), device=dev)
out, info = {}, {"src": args.src, "layers": {}}
for L in Ls:
    h = bench.resid[L][fi].float().to(dev)
    T, Ms, cs = [], [], []
    for _, kind, S in srcs:
        if kind == "map":
            mx, W, my = (x.to(dev) for x in S[L])
            T.append(((h - mx) @ W + my)[:, None])
            Ms.append(W.T[None].cpu())
            cs.append((my - mx @ W)[None].cpu())
        else:
            A = S[L].to(dev)
            T.append(torch.einsum("pn,kfn->pkf", h, A))
            Ms.append(S[L])
            cs.append(torch.zeros(A.shape[0], A.shape[1]))
            del A
    T = torch.cat(T, 1)
    lg = expert_logits(model.unembed, T)
    sc = expert_pair_pass_scores(lg, pairs, excluded_mask_Bool_V=excl, k=CFG.pass_k)
    with torch.enable_grad():
        lw = fit_expert_weights(lg, pairs, excluded_mask_Bool_V=excl,
                                initial_weightings=build_search_starting_points(sc, top_n=min(CFG.top_n, len(sc))),  # top_n <= K (2-term case)
                                condense_config=CFG)
    w = lw.weights_E.float()
    Mcat, ccat = torch.cat(Ms, 0), torch.cat(cs, 0)
    out[L] = {"M": torch.einsum("k,kfn->fn", w, Mcat), "c": torch.einsum("k,kf->f", w, ccat)}
    info["layers"][str(L)] = dict(w=w.tolist(), single=sc.tolist(), train=lw.train_macro_pass, init=lw.init_name,
                                  step=lw.step)
    print(f"[xj {args.name}] L{L}: K={len(w)} single {[round(x, 3) for x in sc.tolist()]} -> train "
          f"{lw.train_macro_pass:.3f} ({lw.init_name}, step {lw.step})", flush=True)
    del lg, T
    torch.cuda.empty_cache()


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


rows = []
for L in Ls:
    h = bench.resid[L][fi].float().to(dev)
    v = unit(h @ out[L]["M"].to(dev).T + out[L]["c"].to(dev))
    rows += readout_rows("x", L, bench, v, model, filtered=True, items=fi, chunk_size=64)
info["fit_p10"], info["fit_task"] = ev.pass_at_k(rows, 10)
info["fit_p1"], _ = ev.pass_at_k(rows, 1)
od = f"{R}/runs/ao1/{m}/xj"
os.makedirs(od, exist_ok=True)
torch.save(out, f"{od}/{args.name}.pt")
json.dump(info, open(f"{od}/{args.name}.json", "w"), indent=1)
print(f"[xj {args.name}] FIT p10 {info['fit_p10']:.4f} p1 {info['fit_p1']:.4f} {info['fit_task']}", flush=True)
