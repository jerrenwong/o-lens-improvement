"""ex2 S4 (one-scalar contrasts, chosen on FIT) and S5 (FIT-item bootstrap of the authors' weight search). GPU,
lite model (final norm + LM head). FIT rows of the bench only; nothing here touches DEV.

FIT score = the scorer's metric on the FIT rows (auto/evaluate.py pass_at_k over olens.readout.readout_rows,
filtered, pass@10, best rank over layers, item-weighted macro over tasks).

S4 lenses per fit (basis E_0..E_7, Jbar; a_X = <X, Jbar>_F / <Jbar, Jbar>_F from runs/ex2/gram.json):
  single : E_{e*(layer)} - alpha a_{E_e*} Jbar, e*(layer) = the expert with the best stand-alone FIT pair score
           (jpp_lens expert_pair_pass_scores = the authors' select_best start); alpha in ALPHAS chosen on FIT
  content: mean of the label-free content experts (ex2_common.content_experts) - alpha a Jbar (9B fits, 2b_A)
  P      : sum_e max(w_e,0) E_e - alpha a_P Jbar
  sum0   : weights refit on FIT with sum_e w_e = 0 (pure deviations): a copy of jpp_lens fit_expert_weights with
           the weights projected onto {sum = 0} at every step; starts = projected select_best and mean_top
           (the uniform start projects to 0). No scalar.
S5: for the main fits, 200 bootstrap resamples of the FIT items (with replacement, within task), each refit
with the authors' search (fit_expert_weights, build_search_starting_points, CondenseConfig defaults) on cached
per-expert logits (expert_logits) -> weights per layer. Plus a refit on the original items (reproduction check).
Writes runs/ex2/s4_<fit>.json, runs/ex2/s5_<fit>.json
    python scripts/ex2/s45_fit.py --model 9b [--boot 200] [--fits 9b_64a,...] [--boot-fits 9b_64a]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
sys.path.insert(0, "/workspace/o-lens-improvement/src")
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/perf1")
sys.path.insert(0, "/root/jpp_lens/src")
from ex2_common import BENCH, FITS, LAYERS, MAIN, MODEL, OUT, R, content_experts, load_experts, load_matrix_dict, load_weights  # noqa: E402
from lite_model import get_lite_model  # noqa: E402
from olens.readout import Bench, readout_rows  # noqa: E402
from workspace_lens.fitting.condense_experts.fitter import (  # noqa: E402
    CondenseConfig, build_search_starting_points, fit_expert_weights)
from workspace_lens.fitting.condense_experts.rank_objective import (  # noqa: E402
    build_pair_table, combined_logits, exact_macro_pass, expert_logits, expert_pair_pass_scores, hit_margins,
    soft_macro_pass)

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["9b", "2b"], required=True)
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--fits", default=None)
ap.add_argument("--boot-fits", default=None)
ap.add_argument("--skip-s4", action="store_true")
ap.add_argument("--boot-layers", default=None, help="comma list: bootstrap only these layers (parallel runs)")
ap.add_argument("--boot-tag", default="", help="suffix of the s5 output file")
args = ap.parse_args()
m = args.model
torch.set_grad_enabled(False)
torch.set_num_threads(48)
dev = "cuda"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CFG = CondenseConfig()

spec = importlib.util.spec_from_file_location("ex2_ev", f"{R}/auto/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

bench = Bench.load(BENCH[m])
fi = bench.index("fit")
model = get_lite_model(MODEL[m])
V_SIZE = model._lm_head.weight.shape[0]
excl_mask = torch.zeros(V_SIZE, dtype=torch.bool, device=dev)
excl_mask[bench.excluded_ids.to(dev)] = True
gram = json.load(open(f"{OUT}/gram.json"))["fits"]
CONTENT = content_experts()
slugs = [bench.eval_slugs[i] for i in fi]
cands = [bench.candidates[i] for i in fi]
fits = args.fits.split(",") if args.fits else [f for f, c in FITS.items() if c["m"] == m and f != "9b_w2"]
boot_fits = args.boot_fits.split(",") if args.boot_fits else [MAIN[m]]


def fit_score(vecs_by_layer):
    rows = []
    for L, v in vecs_by_layer.items():
        rows += readout_rows("x", L, bench, v, model, filtered=True, items=fi, chunk_size=64)
    p10, per = ev.pass_at_k(rows, 10)
    return p10


def fit_sum0(logits_PEV, pairs, starts):
    """jpp_lens fit_expert_weights with the weights kept on {sum_e w_e = 0} (projection before normalising)."""
    E = logits_PEV.shape[1]
    Pm = (torch.eye(E) - torch.ones(E, E) / E).to(logits_PEV.device)
    best = None
    for name, s in starts.items():
        v0 = Pm @ s.float().to(logits_PEV.device)
        if v0.norm() < 1e-8:
            continue
        v = (v0 / v0.norm()).clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=CFG.learning_rate)
        for step in range(CFG.steps + 1):
            with torch.enable_grad():
                u = Pm @ v
                u = u / u.norm()
                mg, sp = hit_margins(combined_logits(logits_PEV, u), pairs, excluded_mask_Bool_V=excl_mask, k=CFG.pass_k)
                sur = soft_macro_pass(mg, sp, pairs, temperature=CFG.temperature)
                if step % CFG.exact_every == 0 or step == CFG.steps:
                    cand = (exact_macro_pass(mg.detach(), pairs), float(sur.detach()), u.detach().cpu().clone(), name, step)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
                if step == CFG.steps:
                    break
                opt.zero_grad()
                (-sur).backward()
                opt.step()
    return dict(w=best[2].tolist(), train=best[0], init=best[3], step=best[4])


for fit in fits:
    t0 = time.time()
    has4 = os.path.exists(f"{OUT}/s4_{fit}.json")
    if has4 and (fit not in boot_fits or os.path.exists(f"{OUT}/s5_{fit}{args.boot_tag}.json")):
        print(f"[s45] skip {fit} (done)", flush=True)
        continue
    E, _, _, _ = load_experts(fit)
    Jb = load_matrix_dict(FITS[fit]["pooled"], LAYERS[m])
    W = load_weights(fit)
    out4 = {"alphas": ALPHAS, "layers": {}, "fit_scores": {}}
    T, LG = {}, {}
    for L in LAYERS[m]:
        h = bench.resid[L][fi].float().to(dev)
        B = torch.cat([E[L].float(), Jb[L][None]])  # cpu: the GPU is shared
        T[L] = torch.einsum("pn,kfn->pkf", h.cpu(), B).to(dev)  # [P, 9, F] transports of E_0..E_7, Jbar
        LG[L] = expert_logits(model.unembed, T[L][:, :8].contiguous()).cpu()  # [P, 8, V] as the fitter caches it
        del B
    pairs_all = build_pair_table(cands, slugs, list(range(len(fi))), device=dev)

    def vecs(coefs):  # coefs {L: 9-vector over (E0..E7, Jbar)}
        return {L: torch.einsum("k,pkf->pf", torch.tensor(c, dtype=torch.float32, device=dev), T[L]) for L, c in coefs.items()}

    def proj(L, c8):  # <sum c E, Jbar> / <Jbar, Jbar>
        G = np.array(gram[fit][str(L)]["G"])
        return float(np.dot(c8, G[:8, 8]) / G[8, 8])

    if not (args.skip_s4 or has4):
        base = {}
        for L in LAYERS[m]:
            lgL = LG[L].to(dev)
            sc = expert_pair_pass_scores(lgL, pairs_all, excluded_mask_Bool_V=excl_mask, k=CFG.pass_k)
            w = np.array(W[L])
            estar = int(torch.argmax(sc))  # ties -> lowest index (argmax returns first)
            c_single = np.eye(8)[estar]
            cont = CONTENT.get(fit, {}).get(L)
            c_cont = (np.isin(np.arange(8), cont).astype(float) / max(1, len(cont))) if cont else None
            c_P = np.clip(w, 0, None)
            starts = build_search_starting_points(sc, top_n=CFG.top_n)
            s0 = fit_sum0(lgL, pairs_all, {k: v for k, v in starts.items() if k != "uniform"})
            base[L] = dict(single_scores=sc.tolist(), estar=estar, c_single=c_single.tolist(),
                           a_single=proj(L, c_single), content=cont,
                           c_content=None if c_cont is None else c_cont.tolist(),
                           a_content=None if c_cont is None else proj(L, c_cont),
                           c_P=c_P.tolist(), a_P=proj(L, c_P), sum0=s0, w=w.tolist(), a_L=proj(L, w))
            del lgL
            torch.cuda.empty_cache()
            print(f"[s4] {fit} L{L}: e*={estar} content={cont} sum0 train {s0['train']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        out4["layers"] = {str(k): v for k, v in base.items()}
        fs = out4["fit_scores"]
        fs["L"] = fit_score(vecs({L: list(base[L]["w"]) + [0.0] for L in LAYERS[m]}))
        fs["pooled"] = fit_score(vecs({L: [0.0] * 8 + [1.0] for L in LAYERS[m]}))
        fs["sum0"] = fit_score(vecs({L: list(base[L]["sum0"]["w"]) + [0.0] for L in LAYERS[m]}))
        for fam in ("single", "content", "P"):
            if fam == "content" and base[LAYERS[m][0]]["c_content"] is None:
                continue
            fs[fam] = {}
            for a in ALPHAS:
                fs[fam][str(a)] = fit_score(vecs({L: list(base[L][f"c_{fam}"]) + [-a * base[L][f"a_{fam}"]]
                                                  for L in LAYERS[m]}))
            best_a = max(ALPHAS, key=lambda a: (fs[fam][str(a)], -a))
            out4[f"alpha_{fam}"] = best_a
            print(f"[s4] {fit} {fam}: FIT p10 by alpha {fs[fam]} -> alpha {best_a}", flush=True)
        print(f"[s4] {fit} FIT p10 L {fs['L']:.4f} pooled {fs['pooled']:.4f} sum0 {fs['sum0']:.4f}", flush=True)
        json.dump(out4, open(f"{OUT}/s4_{fit}.json", "w"), indent=1)

    if fit in boot_fits:
        rng = np.random.default_rng(0)
        by_task = {}
        for i, s in enumerate(slugs):
            by_task.setdefault(s, []).append(i)
        draws = [list(range(len(fi)))] + [
            sorted(int(x) for s in sorted(by_task) for x in rng.choice(by_task[s], len(by_task[s]), replace=True))
            for _ in range(args.boot)]
        out5 = {"draws": draws, "weights": {}, "train": {}, "stored": {str(L): W[L] for L in LAYERS[m]}}
        for L in (LAYERS[m] if args.boot_layers is None else [int(x) for x in args.boot_layers.split(",")]):
            lgL = LG[L].to(dev)
            ws, tr = [], []
            for b, idx in enumerate(draws):
                it = torch.tensor(idx, device=dev)
                lg = lgL[it]
                pr = build_pair_table(cands, slugs, idx, device=dev)
                sc = expert_pair_pass_scores(lg, pr, excluded_mask_Bool_V=excl_mask, k=CFG.pass_k)
                with torch.enable_grad():
                    lw = fit_expert_weights(lg, pr, excluded_mask_Bool_V=excl_mask,
                                            initial_weightings=build_search_starting_points(sc, top_n=CFG.top_n),
                                            condense_config=CFG)
                ws.append(lw.weights_E.tolist())
                tr.append(lw.train_macro_pass)
                del lg
                if b == 0:
                    d = float(np.abs(np.array(ws[0]) - np.array(W[L])).max())
                    print(f"[s5] {fit} L{L} reproduction: max|w - stored| = {d:.2e}, train {tr[0]:.4f}", flush=True)
                if b % 50 == 0:
                    print(f"[s5] {fit} L{L} draw {b} ({time.time() - t0:.0f}s)", flush=True)
            del lgL
            torch.cuda.empty_cache()
            out5["weights"][str(L)] = ws
            out5["train"][str(L)] = tr
            json.dump(out5, open(f"{OUT}/s5_{fit}{args.boot_tag}.json", "w"))
    del T, LG, E, Jb
    torch.cuda.empty_cache()
print("[s45] done", flush=True)
