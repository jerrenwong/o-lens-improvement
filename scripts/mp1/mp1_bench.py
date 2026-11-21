"""mp1 step 3: benchmark analysis on the ORIGINAL 9B benchmark, FIT + DEV prompts only (never the test list; text and
readout positions from the viewer export as auto/qual3.py; scoring = olens.readout.readout_rows + evaluate.pass_at_k as
scripts/why1_analysis3.py).  Mean residual over all prompt tokens 0..readout pos at the 7 readout layers
(hook on model.layers[l], same convention as al2 / M_mean; checked against bench.resid at the readout position).
Readers on DEV: (1) J++ h_readout (benchmark convention), (2) J++ h_mean, (3) best-of-both (2x guesses),
(4) unit(J h_r) + lam unit(J h_mean), lam in {0.3, 1.0} chosen on FIT, (5) unit(J h_r) + lam unit(M_mean h_mean),
lam in {0.1, 0.3, 1.0} chosen on FIT, (6) M_mean h_mean alone (reference). Paired bootstrap (items, stratified by task)."""
import json, random, sys, time
from collections import defaultdict
import torch
R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402
from olens.readout import Bench, readout_rows  # noqa: E402
sys.path += [f"{R}/auto", f"{R}/auto/candidates"]
from evaluate import pass_at_k  # noqa: E402
from e2_lib import load_maps  # noqa: E402

M = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
OUT = f"{R}/.hf27/mp1"
dev = "cuda"
torch.set_grad_enabled(False)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


model = get_hf_model(M, attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
tok = model.tokenizer
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
MM = load_maps(f"{OUT}/M_mean.pt", dev)
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
split = json.load(open(f"{R}/auto/split_9b.json"))
FIT_K, DEV_K = set(map(tuple, split["fit"])), set(map(tuple, split["dev"]))
del split  # the test list is never used
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
dev_idx = sorted(pos[k] for k in DEV_K)
fit_idx = sorted(pos[k] for k in FIT_K)
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json"))
RV = rd["vocab"]
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in FIT_K | DEV_K}


def item_ids(i):
    k = (bench.eval_slugs[i], bench.item_names[i])
    assert k in FIT_K or k in DEV_K
    it = meta[k]
    text = "".join(RV[x] for x in it["toks"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ok = len(ids) == len(it["toks"]) and all(tok.decode([a]) == RV[b] for a, b in zip(ids, it["toks"]))
    return ids, it["pos"], ok


def means(idx):
    """{l: [n, d]} mean over tokens 0..pos; also cos(hook at pos, bench.resid) min."""
    out = {l: torch.zeros(len(idx), model.d_model, device=dev) for l in LAYERS}
    cmin, ok_all, ntok = 1.0, 0, []
    for j, i in enumerate(idx):
        ids, p, ok = item_ids(i)
        ok_all += ok
        x = torch.tensor([ids[:p + 1]], device=dev)
        rec = {}
        hk = [model.layers[l].register_forward_hook(
            (lambda l: lambda m, a, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[0].float()))(l))
            for l in LAYERS]
        try:
            model.forward(x)
        finally:
            for h in hk:
                h.remove()
        ntok.append(p + 1)
        for l in LAYERS:
            out[l][j] = rec[l].mean(0)
            cmin = min(cmin, torch.cosine_similarity(rec[l][p], bench.resid[l][i].to(dev).float(), 0).item())
    return out, dict(n=len(idx), tokens_ok=ok_all, min_cos_hook_vs_bench=cmin,
                     mean_tokens=sum(ntok) / len(ntok), max_tokens=max(ntok))


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def rows_of(name, vecs, idx, filtered=True):
    rows = []
    for l, v in vecs.items():
        rows += readout_rows(name, l, bench, v, model, filtered=filtered, items=idx, chunk_size=1)
    return rows


def item_scores(rows, k):
    best = {}
    for r in rows:
        key = (r["eval"], r["item"], r["intermediate"])
        best[key] = min(best.get(key, r["rank"]), r["rank"])
    it = defaultdict(list)
    for (e, n, _), rk in best.items():
        it[(e, n)].append(float(rk <= k))
    return {key: sum(v) / len(v) for key, v in it.items()}, best


def summ(rows):
    p10, per10 = pass_at_k(rows, 10)
    p1, per1 = pass_at_k(rows, 1)
    return dict(pass10=round(p10, 4), pass1=round(p1, 4), per10={k: round(v, 4) for k, v in per10.items()},
                per1={k: round(v, 4) for k, v in per1.items()})


def macro(sc, keys):
    t = defaultdict(list)
    for k in keys:
        t[k[0]].append(sc[k])
    return sum(sum(v) / len(v) for v in t.values()) / len(t)


def boot(a, b, n=4000, seed=0):
    """paired bootstrap of macro(a) - macro(b), items resampled within task. a, b: {item_key: score}."""
    keys = sorted(a)
    bytask = defaultdict(list)
    for k in keys:
        bytask[k[0]].append(k)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        tot = 0.0
        for t, ks in bytask.items():
            s = [rng.choice(ks) for _ in ks]
            tot += sum(a[k] - b[k] for k in s) / len(s)
        diffs.append(tot / len(bytask))
    diffs.sort()
    return round(macro(a, keys) - macro(b, keys), 4), round(diffs[int(.025 * n)], 4), round(diffs[int(.975 * n)], 4)


def ci1(a, n=4000, seed=0):
    keys = sorted(a)
    z = {k: 0.0 for k in keys}
    d, lo, hi = boot(a, z, n, seed)
    return d, lo, hi


res = {}
VEC = {}
for sname, idx in (("fit", fit_idx), ("dev", dev_idx)):
    si = torch.as_tensor(idx, dtype=torch.long)
    hr = {l: bench.resid[l][si].to(dev).float() for l in LAYERS}
    hm, chk = means(idx)
    res[f"check_{sname}"] = chk
    log(sname, chk)
    jr = {l: hr[l] @ JPP[l].T for l in LAYERS}
    jm = {l: hm[l] @ JPP[l].T for l in LAYERS}
    mm = {l: (hm[l] - MM[l][0]) @ MM[l][1] + MM[l][2] for l in LAYERS}
    V = {"1_Jread": jr, "2_Jmean": jm, "6_Mmean_alone": mm}
    for lam in (0.3, 1.0):
        V[f"4_JJ_lam{lam}"] = {l: unit(jr[l]) + lam * unit(jm[l]) for l in LAYERS}
    for lam in (0.1, 0.3, 1.0):
        V[f"5_JM_lam{lam}"] = {l: unit(jr[l]) + lam * unit(mm[l]) for l in LAYERS}
    VEC[sname] = (idx, V, hr, hm)
# lam choice on FIT
idx, V, _, _ = VEC["fit"]
fitp = {n: summ(rows_of(n, v, idx))["pass10"] for n, v in V.items()}
res["fit_pass10"] = fitp
lam4 = max((0.3, 1.0), key=lambda z: (fitp[f"4_JJ_lam{z}"], -z))
lam5 = max((0.1, 0.3, 1.0), key=lambda z: (fitp[f"5_JM_lam{z}"], -z))
res["lam4"], res["lam5"] = lam4, lam5
log("fit", fitp, "lam4", lam4, "lam5", lam5)
# DEV
idx, V, hr, hm = VEC["dev"]
R_ = {n: rows_of(n, v, idx) for n, v in V.items()}
R_["3_best_of_both"] = R_["1_Jread"] + R_["2_Jmean"]
R_["4_JJ"] = R_[f"4_JJ_lam{lam4}"]
R_["5_JM"] = R_[f"5_JM_lam{lam5}"]
res["dev"] = {n: summ(r) for n, r in R_.items()}
for n in ("1_Jread", "2_Jmean", "3_best_of_both", "4_JJ", "5_JM", "6_Mmean_alone"):
    log("dev", n, res["dev"][n])
res["dev_unf"] = {n: round(pass_at_k(rows_of(n, V[n], idx, filtered=False), 10)[0], 4)
                  for n in ("1_Jread", "2_Jmean")}
S10 = {n: item_scores(r, 10)[0] for n, r in R_.items()}
S1 = {n: item_scores(r, 1)[0] for n, r in R_.items()}
res["ci"] = {n: dict(p10=ci1(S10[n]), p1=ci1(S1[n])) for n in ("1_Jread", "2_Jmean", "3_best_of_both", "4_JJ", "5_JM",
                                                                 "6_Mmean_alone")}
res["boot_vs_1"] = {n: dict(p10=boot(S10[n], S10["1_Jread"]), p1=boot(S1[n], S1["1_Jread"]))
                    for n in ("2_Jmean", "3_best_of_both", "4_JJ", "5_JM")}
log("boot", res["boot_vs_1"])
# association: concept-level best rank changes
best = {n: item_scores(R_[n], 10)[1] for n in ("1_Jread", "2_Jmean", "4_JJ", "5_JM")}
ch = []
for key, r1 in sorted(best["1_Jread"].items()):
    if key[0] != "association":
        continue
    o = dict(item=key[1], concept=key[2], r1=r1, r2=best["2_Jmean"][key], r4=best["4_JJ"][key], r5=best["5_JM"][key])
    if any((o[x] <= 10) != (r1 <= 10) for x in ("r2", "r4", "r5")):
        ch.append(o)
res["assoc_changes"] = ch
res["assoc_n_concepts"] = sum(1 for k in best["1_Jread"] if k[0] == "association")
# per-layer association hits
res["assoc_layer_hits10"] = {n: {l: sum(r["rank"] <= 10 for r in R_[n] if r["eval"] == "association" and r["layer"] == l)
                                 for l in LAYERS} for n in ("1_Jread", "2_Jmean", "6_Mmean_alone")}
# qualitative: 8 dev association prompts, top-8 at L12, L20
Vn = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
exc = torch.zeros(Vn, dtype=torch.bool, device=dev)
exc[bench.excluded_ids.to(dev)] = True
assoc = [j for j, i in enumerate(idx) if bench.eval_slugs[i] == "association"][:8]
qual = []
for j in assoc:
    i = idx[j]
    k = (bench.eval_slugs[i], bench.item_names[i])
    it = meta[k]
    rec = dict(item=k[1], text="".join(RV[x] for x in it["toks"][max(0, it["pos"] - 50):it["pos"] + 1]),
               concepts=list(bench.candidates[i].keys()), L={})
    for l in (12, 20):
        rec["L"][l] = {}
        for n, v in (("J_read", V["1_Jread"][l][j:j + 1]), ("J_mean", V["2_Jmean"][l][j:j + 1]),
                     ("M_mean", V["6_Mmean_alone"][l][j:j + 1])):
            lg = model.unembed(v).float()[0, :Vn].masked_fill(exc, float("-inf"))
            rk = {c: int((lg > lg[torch.as_tensor(cid, device=dev).view(-1)].max()).sum()) + 1
                  for c, cid in bench.candidates[i].items()}
            rec["L"][l][n] = dict(top=[strs[t].strip() for t in lg.topk(8).indices.tolist()], ranks=rk)
    qual.append(rec)
res["qual"] = qual
json.dump(res, open(f"{OUT}/bench_results.json", "w"), indent=1, ensure_ascii=False)
log("BENCH_DONE")
