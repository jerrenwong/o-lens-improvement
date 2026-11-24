"""W7 output diagnostics (fit + dev bench rows only; never test).
 (A) maps read alone, NO filter, dev rows x readout layers: share of top-1 that is a non-word (nonword.py rule:
     no letter/digit, whitespace-only, special) and share not allowed by the w7 target rule (adds "@&<>"), and the mean
     number of DISTINCT normal forms (nf7: strip, drop one marker, lower) among the top 10.  Maps: ridge, ce (w2 k10),
     nomerge, merge; for topic, spec and the A-lens average unit(unit(a)+unit(b)).
 (B) candidate coverage: over fit+dev concepts, is the canonical token of nf7(concept) (w7 rule) one of the concept's
     accepted candidate ids?  And a direct check of the effect: ranks of A-lens-alone (merge, ce) with the canonical
     token ADDED to the candidates vs as scored, under f1 (token ranks) and f4 (group ranks); proxy = mean over
     (concept, layer) of rank <= 10 / == 1.
 (C) 10 fit-split prompts: top 10 (unfiltered) of the A-lens average, ce vs merge, at the best layer (proxy f4 top-10 of
     merge on fit rows).
Usage: python w7_outdiag.py <model>"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402
from olens.readout import Bench, candidate_ranks, group_candidate_ranks  # noqa: E402

_fs = importlib.util.spec_from_file_location("w7_filters", f"{R}/auto/filters.py")
FM = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(FM)
torch.set_grad_enabled(False)
M = sys.argv[1]
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
BENCH = {"9b": "artifacts/bench/qwen3.5-9b.pt", "2b": "artifacts/bench/qwen3.5-2b.pt"}[TAG]
dev = "cuda"
model = get_lite_model(M)
tok = model.tokenizer
V = model._lm_head.weight.shape[0]
strs = FM.decode_all(tok, V)
SPEC = FM.special_ids(tok)
special = set(SPEC) | {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
nonword = torch.tensor([not re.search(r"[^\W_]", s) or s.strip() == "" for s in strs], device=dev)
nonword[[i for i in special if i < V]] = True
notallowed = torch.tensor([not (any(c.isalnum() for c in s) and i not in SPEC and not any(c in s for c in "@&<>"))
                           for i, s in enumerate(strs)], device=dev)


def nf7(s):
    s = s.strip()
    if s[:1] in FM.MARKERS:
        s = s[1:]
    return s.lower()


NF = [nf7(s) for s in strs]
k2i = {}
NFID = torch.tensor([k2i.setdefault(n, len(k2i)) for n in NF], device=dev)
exact, space = {}, {}
for i, s in enumerate(strs):
    if not notallowed[i]:
        if s == NF[i]:
            exact.setdefault(NF[i], i)
        elif s == " " + NF[i]:
            space.setdefault(NF[i], i)


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def load(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


def emap(m, h, l):
    mx, W, my = m[l]
    return (h - mx) @ W + my


bench = Bench.load(f"{R}/{BENCH}")
FL = FM.Filters(tok, V, [bench])
F4 = FL.get("f4")
split = json.loads(Path(f"{R}/auto/split_{TAG}.json").read_text())
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
idx = {s: torch.tensor(sorted(pos[tuple(k)] for k in split[s])) for s in ("fit", "dev")}
H = {s: {l: bench.resid[l][ix].to(dev).float() for l in LAYERS} for s, ix in idx.items()}
pairs = {"ridge": (f"{R}/.hf27/ask3/E_{TAG}_t3.pt", f"{R}/.hf27/ask5/E_{TAG}_s5.pt"),
         "ce": (f"{R}/.hf27/w2/E_{TAG}_topic_ce_k10.pt", f"{R}/.hf27/w2/E_{TAG}_spec_ce_k10.pt")}
for mode in ("all", "nomerge", "mergeall", "merge"):
    pairs[mode] = (f"{R}/.hf27/w7/E_{TAG}_topic_{mode}_k10.pt", f"{R}/.hf27/w7/E_{TAG}_spec_{mode}_k10.pt")
ONLY_A = sys.argv[2:]  # e.g. nomerge_k1 nomerge_k10_t0.5 ... -> section (A) only, A-lens average, for these map suffixes
if ONLY_A:
    pairs = {v: (f"{R}/.hf27/w7/E_{TAG}_topic_{v}.pt", f"{R}/.hf27/w7/E_{TAG}_spec_{v}.pt") for v in ONLY_A}
MAPS = {n: (load(a), load(b)) for n, (a, b) in pairs.items()}


def lens_logits(name, s, l, which="alens"):
    Et, Es = MAPS[name]
    a, b = unit(emap(Et, H[s][l], l)), unit(emap(Es, H[s][l], l))
    v = {"topic": a, "spec": b, "alens": unit(a + b)}[which]
    return model.unembed(v).float()


out = {}
print(f"\n(A) {TAG} dev rows, unfiltered, mean over layers")
print(f"{'maps':8s} {'read':6s} | top1 non-word | top1 not-allowed(w7) | distinct nf7 in top10 | top1 markup/@")
mk = torch.tensor([any(c in s for c in "@&<>") for s in strs], device=dev)
for name in pairs:
    for which in ("topic", "spec", "alens"):
        nw, na, dn, mm = [], [], [], []
        for l in LAYERS:
            L = lens_logits(name, "dev", l, which)
            t10 = L.topk(10, -1).indices
            nw.append(nonword[t10[:, 0]].float().mean().item())
            na.append(notallowed[t10[:, 0]].float().mean().item())
            mm.append(mk[t10[:, 0]].float().mean().item())
            nfs = NFID[t10]
            dn.append(sum(len(set(r)) for r in nfs.tolist()) / len(nfs))
        r = dict(nonword1=sum(nw) / len(nw), notallowed1=sum(na) / len(na), distinct_nf10=sum(dn) / len(dn),
                 markup_at1=sum(mm) / len(mm))
        out[f"A/{name}/{which}"] = r
        if ONLY_A and which != "alens":
            continue
        print(f"{name:8s} {which:6s} | {r['nonword1']:.3f}         | {r['notallowed1']:.3f}                | "
              f"{r['distinct_nf10']:.2f}                  | {r['markup_at1']:.3f}", flush=True)

if ONLY_A:
    json.dump(out, open(f"{R}/runs/w7/outdiagA_sweep_{TAG}.json", "w"), indent=1)
    sys.exit(0)
# (B) candidate coverage
print(f"\n(B) {TAG} candidate coverage of the canonical token (fit+dev concepts)")
stat = dict(n=0, a_in=0, a_out=0, b_in=0, b_out=0, none=0)
canon = {}
for s in ("fit", "dev"):
    for i in idx[s].tolist():
        for inter, ids in bench.candidates[i].items():
            n = nf7(inter)
            stat["n"] += 1
            c = exact.get(n)
            kind = "a"
            if c is None:
                c, kind = space.get(n), "b"
            if c is None:
                stat["none"] += 1
                continue
            canon[(i, inter)] = c
            stat[f"{kind}_{'in' if c in ids else 'out'}"] += 1
print(f"  concepts {stat['n']}: canonical = exact lower-case no-space token: in candidates {stat['a_in']}, NOT in "
      f"{stat['a_out']}; canonical = ' '+nf token: in {stat['b_in']}, not in {stat['b_out']}; no single-token "
      f"spelling of nf7 {stat['none']}")
miss = [(bench.item_names[i], inter, strs[c], [strs[t] for t in bench.candidates[i][inter]][:6])
        for (i, inter), c in canon.items() if c not in bench.candidates[i][inter]]
print("  examples not covered:", miss[:12])
out["B/coverage"] = stat
mask1 = torch.zeros(V, dtype=torch.bool, device=dev); mask1[bench.excluded_ids] = True
mask4 = torch.zeros(V, dtype=torch.bool, device=dev); mask4[F4.excluded_ids] = True
gid = F4.group_ids.to(dev)
print(f"  effect on proxy scores (dev, A-lens alone, mean over concept x layer): rank<=10 / rank==1, as scored -> "
      f"with canonical added to candidates")
for name in ("ridge", "ce", "all", "nomerge", "mergeall", "merge"):
    for lv in ("f1", "f4"):
        acc = {"as": [0, 0, 0], "plus": [0, 0, 0]}
        for l in LAYERS:
            L = lens_logits(name, "dev", l)
            cands = [bench.candidates[i] for i in idx["dev"].tolist()]
            cplus = [{k: sorted(set(v) | ({canon[(i, k)]} if (i, k) in canon else set())) for k, v in c.items()}
                     for i, c in zip(idx["dev"].tolist(), cands)]
            for key, cc in (("as", cands), ("plus", cplus)):
                if lv == "f1":
                    rk = candidate_ranks(L.masked_fill(mask1, -1e9), cc)
                else:
                    rk = group_candidate_ranks(L.masked_fill(mask4, -1e9), cc, gid)
                for d in rk:
                    for r in d.values():
                        acc[key][0] += 1; acc[key][1] += r <= 10; acc[key][2] += r == 1
        a, p = acc["as"], acc["plus"]
        out[f"B/effect/{name}/{lv}"] = dict(as_scored=[a[1] / a[0], a[2] / a[0]], plus=[p[1] / p[0], p[2] / p[0]])
        print(f"  {name:8s} {lv}: {a[1] / a[0]:.3f} / {a[2] / a[0]:.3f} -> {p[1] / p[0]:.3f} / {p[2] / p[0]:.3f}", flush=True)

# (C) qualitative: fit prompts
best, bl = -1, LAYERS[0]
for l in LAYERS:
    L = lens_logits("merge", "fit", l)
    rk = group_candidate_ranks(L.masked_fill(mask4, -1e9), [bench.candidates[i] for i in idx["fit"].tolist()], gid)
    sc = sum(r <= 10 for d in rk for r in d.values())
    if sc > best:
        best, bl = sc, l
print(f"\n(C) {TAG} fit prompts, A-lens average read alone, unfiltered top 10, layer {bl} (best f4 proxy for merge on fit)")
fit_ix = idx["fit"].tolist()
Lc, La, Lm = lens_logits("ce", "fit", bl), lens_logits("all", "fit", bl), lens_logits("merge", "fit", bl)
for j in range(0, len(fit_ix), max(1, len(fit_ix) // 10))[:10]:
    i = fit_ix[j]
    print(f"--- {bench.eval_slugs[i]}/{bench.item_names[i]}  concepts {list(bench.candidates[i])}")
    print(f"  ce   : {[strs[t] for t in Lc[j].topk(10).indices.tolist()]}")
    print(f"  C2all: {[strs[t] for t in La[j].topk(10).indices.tolist()]}")
    print(f"  merge: {[strs[t] for t in Lm[j].topk(10).indices.tolist()]}")
json.dump(out, open(f"{R}/runs/w7/outdiag_{TAG}.json", "w"), indent=1)
