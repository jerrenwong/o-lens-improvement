"""why1: WHY does J++ + 0.5*E help on Qwen3.5-9B?  Analysis only (dev split only, no test/heldout).

Stages (each caches its result in runs/why1/<stage>.json; a retry skips finished stages):
  q1maps  : ridge maps from INPUT EMBEDDINGS ((a) last token, (b) mean of last 16, (c) mean of all prefix tokens)
            to the same elicitation target as e2_fit.py (same val/test/train partition, same ridge rule and grid);
            held-out top-10/top-1 agreement on the 1000 test prefixes, next to E per layer (recomputed).
  q2agree : E with W truncated to its top-r singular directions (bias kept), held-out agreement per layer.
  q4      : (i) model's own next-token top-10 at the prefix's last token (1000 test prefixes); (ii) any of 16
            greedy-continuation steps (first 300 test prefixes); (iii) J++ and E agreement for reference.
  q1bench : standalone scorer (auto/evaluate.py functions; readout_rows chunk_size=1, pass_at_k) on DEV:
            repro J++ and A, then J++ + 0.5*unit(map_x(emb features of the benchmark prompt up to readout pos)).
            Also a FIT-split lam sweep (diagnostic only).
  qual    : top-8 tokens at L12/L20/L28 for J++, E alone, mix (and bag16 map alone) on 12 dev prompts.
"""
import collections
import json
import os
import re
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

from olens.readout import Bench, readout_rows  # noqa: E402
sys.path += [f"{R}/auto", f"{R}/auto/candidates"]  # appended AFTER the packages: auto/candidates/jlens.py shadows jlens
from evaluate import pass_at_k  # noqa: E402  (scorer's own metric)
from e2_lib import ELens, load_maps  # noqa: E402

OUT = f"{R}/runs/why1"
os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
RANKS = [4, 16, 64, 256, 1024]
LAM = 0.5
torch.set_grad_enabled(False)
dev = "cuda"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def done(stage):
    return os.path.exists(f"{OUT}/{stage}.json")


def save(stage, obj):
    with open(f"{OUT}/{stage}.json", "w") as f:
        json.dump(obj, f, indent=1)
    log(f"saved {stage}")


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


model = get_hf_model(M, attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
tok, hf = model.tokenizer, model._hf_model
EMB = hf.get_input_embeddings().weight  # [V_in, d] bf16
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
EMAPS = load_maps(f"{R}/runs/e1/E_9b.pt", dev)  # {l: (mx, W*scale, my)}
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
mask = nonword | func
norm_s = [s.strip().lower() for s in strs]
vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
log(f"model loaded; V={V}, emb rows {EMB.shape[0]}")


def topk_logits(lg, k=10):
    return lg.float()[..., :V].masked_fill(mask, float("-inf")).topk(k, -1).indices


def topk(v, k=10, bs=250):
    out = []
    for i in range(0, v.shape[0], bs):
        out.append(topk_logits(model.unembed(v[i:i + bs].to(dev).float()), k))
    return torch.cat(out)


def agree(top, word):
    return (NORM[top] == NORM[word][:, None]).any(-1)


class Ridge:  # identical to scripts/e2_fit.py
    def __init__(self, X):
        self.mx = X.mean(0)
        self.Xc = (X - self.mx).double()
        s, U = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.s, self.U = s.clamp_min(0), U
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.UtB = self.U.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.U @ (self.UtB / (self.s + c * self.scale)[:, None])).float()

    def predict(self, X, W):
        return (X - self.mx) @ W + self.my


def emb_feats(id_lists):
    """[n, 3, d] fp32: (a) last token, (b) mean of last 16, (c) mean of all."""
    out = torch.zeros(len(id_lists), 3, model.d_model, device=dev)
    for i, ids in enumerate(id_lists):
        e = EMB[torch.tensor(ids, device=EMB.device)].float().to(dev)
        out[i, 0], out[i, 1], out[i, 2] = e[-1], e[-16:].mean(0), e.mean(0)
    return out


FEATS = ["a_last", "b_last16", "c_all"]
data = json.load(open("/root/e1/prefixes.json"))
N = len(data)
va, te, tr = slice(0, NV), slice(NV, NV + NT), slice(NV + NT, N)
need_acts = not all(done(s) for s in ("q1maps", "q2agree", "q4")) or not os.path.exists("/root/why1/bagmaps.pt")
if need_acts:
    A = torch.load("/root/e1/acts.pt")
    Yk = [unit(A["Y"][k].float()) for k in range(3)]
    Ymean = (sum(Yk) / 3)
    del Yk
    word = topk(Ymean, 10)[:, 0]
    log("acts loaded; elicited words computed")

# ---------------------------------------------------------------- Q1 maps
os.makedirs("/root/why1", exist_ok=True)
if not done("q1maps") or not os.path.exists("/root/why1/bagmaps.pt"):
    F = emb_feats([o["ids"] for o in data])
    res, bag = {}, {}
    for j, fname in enumerate(FEATS):
        X = F[:, j].contiguous()
        r = Ridge(X[tr])
        r.setY(Ymean[tr].to(dev))
        sc = [agree(topk(r.predict(X[va], r.W(c))), word[va]).float().mean().item() for c in GRID]
        b = max(range(len(GRID)), key=lambda i: (sc[i], i))
        W = r.W(GRID[b])
        t = topk(r.predict(X[te], W))
        res[fname] = dict(c=GRID[b], val=sc, test10=agree(t, word[te]).float().mean().item(),
                          test1=agree(t[:, :1], word[te]).float().mean().item())
        bag[fname] = (r.mx.cpu(), W.cpu(), r.my.cpu())
        log(fname, res[fname])
        del r
        torch.cuda.empty_cache()
    for l in LAYERS:
        X = A["X"][l][te].float().to(dev)
        mx, W, my = EMAPS[l]
        t = topk((X - mx) @ W + my)
        res[f"E_L{l}"] = dict(test10=agree(t, word[te]).float().mean().item(),
                              test1=agree(t[:, :1], word[te]).float().mean().item())
    tp = topk(Ymean[tr].mean(0, keepdim=True).to(dev)).expand(NT, -1)
    res["prior_only"] = dict(test10=agree(tp, word[te]).float().mean().item())
    torch.save(bag, "/root/why1/bagmaps.pt")
    save("q1maps", res)
    del F

# ---------------------------------------------------------------- Q2 agreement
SVD = {}


def svd_layer(l):
    if l not in SVD:
        U, S, Vh = torch.linalg.svd(EMAPS[l][1])
        SVD[l] = (U, S, Vh)
    return SVD[l]


if not done("q2agree"):
    res = {}
    anyr = {r: torch.zeros(NT, dtype=torch.bool, device=dev) for r in RANKS + [4096]}
    for l in LAYERS:
        X = A["X"][l][te].float().to(dev)
        mx, W, my = EMAPS[l]
        U, S, Vh = svd_layer(l)
        res[f"L{l}_energy"] = {r: (S[:r].pow(2).sum() / S.pow(2).sum()).item() for r in RANKS}
        # energy of the actual map output (variance of (X-mx)@W captured by top-r directions)
        for r in RANKS + [4096]:
            Wr = (U[:, :r] * S[:r]) @ Vh[:r] if r < 4096 else W
            h = agree(topk((X - mx) @ Wr + my), word[te])
            anyr[r] |= h
            res[f"L{l}_r{r}"] = h.float().mean().item()
        log(f"q2 L{l}", {r: round(res[f'L{l}_r{r}'], 3) for r in RANKS + [4096]})
    res["any"] = {r: v.float().mean().item() for r, v in anyr.items()}
    save("q2agree", res)

# ---------------------------------------------------------------- Q4
if not done("q4"):
    res = {}
    wt = word[te]
    own10 = torch.zeros(NT, dtype=torch.bool, device=dev)
    own1 = torch.zeros(NT, dtype=torch.bool, device=dev)
    for k in range(NT):
        ids = torch.tensor([data[NV + k]["ids"]], device=dev)
        t = topk_logits(hf(ids).logits[0, -1])[None]
        own10[k] = agree(t, wt[k:k + 1])[0]
        own1[k] = agree(t[:, :1], wt[k:k + 1])[0]
    res["own_next_top10"] = own10.float().mean().item()
    res["own_next_top1"] = own1.float().mean().item()
    log("q4 (i)", res)
    NG = 300
    any16 = torch.zeros(NG, dtype=torch.bool, device=dev)
    gen_word = torch.zeros(NG, dtype=torch.bool, device=dev)
    step_hits = torch.zeros(16, device=dev)
    examples = []
    for k in range(NG):
        ids = torch.tensor([data[NV + k]["ids"]], device=dev)
        gen = []
        for s in range(16):
            lg = hf(ids).logits[0, -1].float()
            t = topk_logits(lg)[None]
            h = agree(t, wt[k:k + 1])[0]
            step_hits[s] += h.float()
            any16[k] |= h
            nxt = lg.argmax().view(1, 1)
            gen.append(int(nxt))
            ids = torch.cat([ids, nxt], 1)
        g = torch.tensor([x for x in gen if x < V], device=dev)
        gen_word[k] = (NORM[g] == NORM[wt[k]]).any()
        if k < 12:
            examples.append(dict(prefix=data[NV + k]["text"][-120:], word=strs[wt[k]].strip(),
                                 continuation=tok.decode(gen), own_next_hit=bool(own10[k]), any16=bool(any16[k])))
    res["own_next_top10_first300"] = own10[:NG].float().mean().item()
    res["any16_top10_first300"] = any16.float().mean().item()
    res["word_generated_in16_first300"] = gen_word.float().mean().item()
    res["per_step_top10_first300"] = (step_hits / NG).tolist()
    # (iii) J++ and E, all 1000 and first 300
    aJ = torch.zeros(NT, dtype=torch.bool, device=dev)
    aE = torch.zeros(NT, dtype=torch.bool, device=dev)
    for l in LAYERS:
        X = A["X"][l][te].float().to(dev)
        hJ = agree(topk(X @ JPP[l].T), wt)
        mx, W, my = EMAPS[l]
        hE = agree(topk((X - mx) @ W + my), wt)
        aJ |= hJ
        aE |= hE
        res[f"JPP_L{l}"], res[f"E_L{l}"] = hJ.float().mean().item(), hE.float().mean().item()
    res["JPP_any"], res["E_any"] = aJ.float().mean().item(), aE.float().mean().item()
    res["JPP_any_first300"], res["E_any_first300"] = aJ[:NG].float().mean().item(), aE[:NG].float().mean().item()
    res["own10_and_JPPany"] = (own10 & aJ).float().mean().item()
    res["JPPany_given_own10"] = (aJ[own10].float().mean().item() if own10.any() else None)
    res["JPPany_given_not_own10"] = aJ[~own10].float().mean().item()
    res["examples"] = examples
    log("q4", {k: v for k, v in res.items() if k != "examples"})
    save("q4", res)

if need_acts:
    del A, Ymean
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- benchmark (standalone scorer)
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
split = json.load(open(f"{R}/auto/split_9b.json"))
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
dev_idx = sorted(pos[tuple(k)] for k in split["dev"])
fit_idx = sorted(pos[tuple(k)] for k in split["fit"])
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json"))
RV = rd["vocab"]
allowed = set(map(tuple, split["fit"])) | set(map(tuple, split["dev"]))
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in allowed}  # fit+dev text only


def item_ids(i):
    k = (bench.eval_slugs[i], bench.item_names[i])
    assert k in allowed
    it = meta[k]
    text = "".join(RV[x] for x in it["toks"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ok = len(ids) == len(it["toks"]) and all(tok.decode([a]) == RV[b] for a, b in zip(ids, it["toks"]))
    return ids, it["pos"], ok


def score(name, vecs, idx):
    rows = {True: [], False: []}
    for layer, v in vecs.items():
        for f in (True, False):
            rows[f] += readout_rows(name, layer, bench, v, model, filtered=f, items=idx, chunk_size=1)
    p10, per = pass_at_k(rows[True], 10)
    p1, _ = pass_at_k(rows[True], 1)
    p10u, _ = pass_at_k(rows[False], 10)
    return dict(pass10=p10, pass1=p1, pass10_unf=p10u, per_task=per)


def resid(idx):
    si = torch.as_tensor(idx, dtype=torch.long)
    return {l: v[si].to(dev).float() for l, v in bench.resid.items()}


if not done("q1bench"):
    bag = torch.load("/root/why1/bagmaps.pt")
    bag = {k: tuple(t.to(dev) for t in v) for k, v in bag.items()}
    res = {"checks": {}}
    for sname, idx in (("dev", dev_idx), ("fit", fit_idx)):
        rr = resid(idx)
        idl, ok_all, cos_min = [], 0, 1.0
        for n_, i in enumerate(idx):
            ids, p, ok = item_ids(i)
            ok_all += ok
            idl.append(ids[:p + 1])
            if sname == "dev":  # hidden-state check: X[l] == hidden_states[l+1] at the readout position
                hs = hf(torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
                for l in (12, 24):
                    cos_min = min(cos_min, torch.cosine_similarity(hs[l + 1][0, p].float(), rr[l][n_], 0).item())
        res["checks"][sname] = dict(n=len(idx), tokens_ok=ok_all, min_cos_hidden_vs_bench=cos_min)
        log("checks", sname, res["checks"][sname])
        F = emb_feats(idl)
        mapout = {f: (F[:, j] - bag[f][0]) @ bag[f][1] + bag[f][2] for j, f in enumerate(FEATS)}
        jv = {l: rr[l] @ JPP[l].T for l in LAYERS}
        lensA = ELens(JPP, EMAPS, "mix", LAM)
        if sname == "dev":
            res["dev_JPP"] = score("why1_jpp", jv, idx)
            log("dev J++", res["dev_JPP"])
            res["dev_A"] = score("why1_A", {l: lensA.transport(rr[l].clone(), l) for l in LAYERS}, idx)
            log("dev A", res["dev_A"])
            for f in FEATS:
                res[f"dev_mix_{f}"] = score(f"why1_{f}", {l: unit(jv[l]) + LAM * unit(mapout[f]) for l in LAYERS}, idx)
                log("dev mix", f, res[f"dev_mix_{f}"])
        else:  # diagnostic lam sweep on FIT (in-sample for nothing: bag maps never saw benchmark text)
            for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
                for f in FEATS + ["E"]:
                    if f == "E":
                        v = {l: ELens(JPP, EMAPS, "mix", lam).transport(rr[l].clone(), l) for l in LAYERS}
                    else:
                        v = {l: unit(jv[l]) + lam * unit(mapout[f]) for l in LAYERS}
                    res[f"fit_{f}_lam{lam}"] = score("x", v, idx)["pass10"]
            log("fit sweep", {k: round(v, 4) for k, v in res.items() if k.startswith("fit_")})
    save("q1bench", res)

# ---------------------------------------------------------------- qualitative table (dev items only)
QUAL = [("association", "office-h", "office"), ("association", "art-h", "art"), ("association", "unesco", "UNESCO"),
        ("association", "pregnant", "pregnant"), ("association", "free-h", "free"), ("association", "field-h", "field"),
        ("multilingual", "irish-opposite-big", "Irish"), ("multilingual", "fr-double-three", "three"),
        ("multilingual", "fr-place-near-far", "place"), ("multilingual", "serbian-opposite-day", "opposite"),
        ("poetry", "couplet-stars-space", "space"), ("poetry", "couplet-jar-car", "car")]
if not done("qual"):
    bag = torch.load("/root/why1/bagmaps.pt")
    bag = {k: tuple(t.to(dev) for t in v) for k, v in bag.items()}
    exc = torch.zeros(V, dtype=torch.bool, device=dev)
    exc[bench.excluded_ids.to(dev)] = True
    devset = set(map(tuple, split["dev"]))
    out = []
    lensA = ELens(JPP, EMAPS, "mix", LAM)
    for e, n, concept in QUAL:
        assert (e, n) in devset, (e, n)
        i = pos[(e, n)]
        cand = bench.candidates[i][concept]
        ids, p, _ = item_ids(i)
        F = emb_feats([ids[:p + 1]])[:, 1]
        bagv = (F - bag["b_last16"][0]) @ bag["b_last16"][1] + bag["b_last16"][2]
        rec = dict(eval=e, item=n, concept=concept, text=tok.decode(ids[max(0, p - 40):p + 1]), layers={}, best={})
        for l in LAYERS:
            h = bench.resid[l][i:i + 1].to(dev).float()
            vs = {"JPP": h @ JPP[l].T, "E": lensA.E(h, l), "mix": lensA.transport(h.clone(), l)}
            if l == 12:
                vs["bag16"] = bagv
            rec["layers"][l] = {}
            for name, v in vs.items():
                lg = model.unembed(v).float()[0].masked_fill(exc, float("-inf"))
                rank = int((lg > lg[cand].max()).sum()) + 1
                rec["best"][name] = min(rec["best"].get(name, 10**9), rank)
                if l in (12, 20, 28) or name == "bag16":
                    rec["layers"][l][name] = dict(rank=rank, top=[strs[j].strip() for j in lg.topk(8).indices.tolist()])
        out.append(rec)
        log("qual", n, rec["best"])
    save("qual", out)
log("ALL DONE")
