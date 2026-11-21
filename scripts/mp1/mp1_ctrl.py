"""mp1 follow-up controls (al2 generated sentences only, no benchmark).
CONTROL 1 (bag of visible words): al2 M3 map (same procedure as mp1_fit.py) fitted from
  (a) bagA = mean INPUT embedding of all tokens of the sentence (no forward pass),
  (b) bagB = mean input embedding of the sentence's DISTINCT tokens minus the 200 most frequent token ids
      (frequency counted on TRAIN sentences; sentences left empty fall back to bagA, counted).
  Each alone, and mixed unit(Jgen h_mean,l) + lam unit(bag) at each readout layer (any-of-7), lam on validation.
CONTROL 2 (ask the model): sentence placed in the few-shot E0 "Passage / One-word topic" prompt (scripts/g1_collect.py),
  model's next-token distribution scored like a lens (full vocab, punctuation filter, any surface form).
Paired comparisons use mp1 ranks.pt (M_mean, mix Jgen+M_mean on the token mean).  M is a learned map, not a Jacobian."""
import json, os, sys, time
from collections import Counter, defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
from al2_common import R, BIG, BIG1, log, retry, load_model
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
OUT = f"{R}/.hf27/mp1"
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [4, 8, 12, 16, 20, 24, 28]
LAMS = [0.1, 0.3, 1.0]
ASK = ("Passage: The tide pulled the little boat away from the dock while the fisherman untangled his nets.\n"
       "One-word topic: fishing\n\n"
       "Passage: The senate voted late on Tuesday to approve the new budget, despite\n"
       "One-word topic: politics\n\n"
       "Passage: Elle a sorti la farine, les œufs et le beurre pour préparer\n"
       "One-word topic: baking\n\n"
       "Passage: Lily saw the old dog shivering in the rain, so she opened the door and\n"
       "One-word topic: kindness\n\n"
       "Passage: ", "\nOne-word topic:")
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style, role, sents = meta["words"], meta["split"], meta["style"], meta["role"], meta["sents"]
N = len(sents)
g = meta["g"].to(dev)
U = torch.load(f"{BIG1}/U.pt").to(dev)
V, d = U.shape
strs = torch.load(f"{BIG1}/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
by_norm = defaultdict(list)
for i, s in enumerate(strs):
    by_norm[s.strip().lower()].append(i)
W_all = sorted(role)
widx = {w: i for i, w in enumerate(W_all)}
forms = [by_norm[w.lower()] for w in W_all]
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
TW = torch.stack([meta["T"][w] for w in W_all]).to(dev).float()
wi = torch.tensor([widx[w] for w in words_s], device=dev)
sty = torch.tensor(style, device=dev)
spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("train", "val", "test")}
TRN, VAL, TE = (spl[r].nonzero()[:, 0] for r in ("train", "val", "test"))
te_sty = sty[TE].cpu()


def ranks_of(vecs, wid, bs=512):
    out = []
    for i in range(0, vecs.shape[0], bs):
        lg = ((vecs[i:i + bs].float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid[i:i + bs]]).max(1).values
        out.append((lg > best[:, None]).sum(1) + 1)
    return torch.cat(out)


def rank_logits(lg, wid):
    lg = lg.float().masked_fill(excl, float("-inf"))
    best = lg.gather(1, FORM[wid]).max(1).values
    return (lg > best[:, None]).sum(1) + 1


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class Ridge:
    def __init__(self, X):
        self.mx = X.mean(0)
        Xc = (X - self.mx).double()
        s, Q = torch.linalg.eigh(Xc.T @ Xc)
        self.s, self.Q, self.Xc = s.clamp_min(0), Q, Xc
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.QtB = self.Q.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.Q @ (self.QtB / (self.s + c * self.scale)[:, None])).float()


def class_stats(X, idx):
    ws = wi[idx]
    uw, inv = torch.unique(ws, return_inverse=True)
    Mn = torch.zeros(len(uw), d, device=dev).index_add_(0, inv, X[idx])
    cnt = torch.zeros(len(uw), device=dev).index_add_(0, inv, torch.ones(len(idx), device=dev))
    Mn /= cnt[:, None]
    Sw = torch.zeros(d, d, device=dev, dtype=torch.float64)
    for i in range(0, len(idx), 16384):
        Rw = X[idx[i:i + 16384]] - Mn[inv[i:i + 16384]]
        Sw += (Rw.T @ Rw).double()
    return uw, Mn, Sw / len(idx)


def whitener(Sw, f):
    s, Q = torch.linalg.eigh(Sw)
    s = s.clamp_min(0)
    return (Q @ torch.diag((s + f * s.mean()) ** -0.5) @ Q.T).float()


def fit_M3(X, tr_idx, Tfn, epsf=EPSF, grid=GRID):
    uw, Mn, Sw = class_stats(X, tr_idx)
    Tm = Tfn(uw)
    for f in epsf:
        P = whitener(Sw, f)
        r = Ridge(Mn @ P); r.setY(Tm)
        for c in grid:
            yield dict(f=f, c=c), (Mn.mean(0), P @ r.W(c), r.my)
        del r, P


def apply(m, x):
    mx, W, my = m
    return (x - mx) @ W + my


def choose(X, tr_idx, val_idx, Tfn=lambda ids: TW[ids], **kw):
    best = None
    for cfg, m in fit_M3(X, tr_idx, Tfn, **kw):
        r = ranks_of(apply(m, X[val_idx]), wi[val_idx])
        z = (round(float((r <= 10).float().mean()), 6), float((r <= 1).float().mean()))
        if best is None or z > best[0]:
            best = (z, cfg, m)
    return best


res = {}
# ------------------------------------------------------------ model: bag features + ask
if not os.path.exists(f"{OUT}/ctrl_feats.pt"):
    model = load_model()
    tok = model.tokenizer
    EMB = model._hf_model.get_input_embeddings().weight
    all_ids = [tok(s, add_special_tokens=False).input_ids for s in sents]
    freq = Counter(t for i in TRN.tolist() for t in all_ids[i])
    top200 = set(t for t, _ in freq.most_common(200))
    bagA = torch.zeros(N, d)
    bagB = torch.zeros(N, d)
    n_empty = 0
    for i, ids in enumerate(all_ids):
        e = EMB[torch.tensor(ids, device=EMB.device)].float()
        bagA[i] = e.mean(0).cpu()
        keep = sorted(set(ids) - top200)
        if keep:
            bagB[i] = EMB[torch.tensor(keep, device=EMB.device)].float().mean(0).cpu()
        else:
            bagB[i] = bagA[i]; n_empty += 1
    log(f"bags done; top-200 examples {[strs[t] for t, _ in freq.most_common(30)]}; bagB empty {n_empty}")
    last = len(model.layers) - 1
    ask_ids = [tok(ASK[0] + s + ASK[1], add_special_tokens=False).input_ids for s in sents]
    TEl = TE.tolist()
    groups = defaultdict(list)
    for i in TEl:
        groups[len(ask_ids[i])].append(i)
    ask_rank = {}
    ask_top = {}

    def one(sub):
        ids = torch.tensor([ask_ids[i] for i in sub], device=dev)
        rec = {}
        h = model.layers[last].register_forward_hook(
            lambda m, a, oo: rec.__setitem__(0, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))
        try:
            model.forward(ids)
        finally:
            h.remove()
        lg = model.unembed(rec[0]).float()[:, :V]
        return rank_logits(lg, wi[torch.tensor(sub, device=dev)]).cpu(), lg.masked_fill(excl, float("-inf")).topk(3, -1).indices.cpu()

    t1 = time.time()
    for L, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), 64):
            sub = idx[b0:b0 + 64]
            r, tp = retry(one, sub)
            for j, i in enumerate(sub):
                ask_rank[i], ask_top[i] = int(r[j]), tp[j].tolist()
    log(f"ask forward {time.time() - t1:.0f}s")
    torch.save(dict(bagA=bagA.half(), bagB=bagB.half(), n_empty=n_empty,
                    ask_rank=torch.tensor([ask_rank[i] for i in TEl]),
                    ask_top=[[strs[t].strip() for t in ask_top[i]] for i in TEl]), f"{OUT}/ctrl_feats.pt")
    del model, EMB
    torch.cuda.empty_cache()
F = torch.load(f"{OUT}/ctrl_feats.pt")
res["bagB_empty"] = F["n_empty"]
ASKR = F["ask_rank"]

# ------------------------------------------------------------ fit bag maps
JG = {int(l): v.to(dev).float() for l, v in
      load_lens_file(f"{R}/runs/w1/jpp_Afull_Wtrain.pt", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}
TR, VR = defaultdict(dict), defaultdict(dict)
bv, bt = {}, {}
for b in ("bagA", "bagB"):
    X = F[b].to(dev).float()
    z, cfg, m = choose(X, TRN, VAL)
    res[f"choice_{b}"] = dict(cfg=cfg, val_top10=z[0])
    bv[b], bt[b] = apply(m, X[VAL]), apply(m, X[TE])
    TR[b] = ranks_of(bt[b], wi[TE]).cpu()
    log(b, res[f"choice_{b}"], "test top10", float((TR[b] <= 10).float().mean()))
    del X, m
for l in READ:
    X = torch.load(f"{OUT}/Xmean_L{l}.pt").to(dev).float()
    jv, jt = X[VAL] @ JG[l].T, X[TE] @ JG[l].T
    del X
    for b in ("bagA", "bagB"):
        for lam in LAMS:
            VR[f"mix{lam}|{b}"][l] = ranks_of(unit(jv) + lam * unit(bv[b]), wi[VAL]).cpu()
            TR[f"mix{lam}|{b}"][l] = ranks_of(unit(jt) + lam * unit(bt[b]), wi[TE]).cpu()
    torch.cuda.empty_cache()


def any7(store):
    return torch.stack([store[l] for l in READ]).min(0).values


def t10(r):
    return float((r <= 10).float().mean())


A = {}
for b in ("bagA", "bagB"):
    vm = {lam: t10(any7(VR[f"mix{lam}|{b}"])) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    res[f"lam_{b}"], res[f"mixval_{b}"] = lam, vm
    A[f"mix Jgen(mean)+{b}"] = any7(TR[f"mix{lam}|{b}"])
    A[b] = TR[b]
P = torch.load(f"{OUT}/ranks.pt", weights_only=False)
assert torch.equal(P["te_rows"], TE.cpu())
PT = P["TR"]
for n, key in (("Jrel last", "Jrel|last"), ("Jgen mean", "Jgen|mean"), ("M_mean", "M_mean"), ("M_all (last)", "M_all"),
               ("mix Jgen(mean)+M_mean", "mix|Jgen+M_mean (mean)"), ("mix Jgen(last)+M_all", "mix|Jgen+M_all (last)")):
    A[n] = any7(PT[key])
for l in (4, 12, 20, 28):
    A[f"M_mean L{l} only"] = PT["M_mean"][l]
A["ask the model"] = ASKR


def acc(r):
    return [round(float((r <= 10).float().mean()), 4), round(float((r <= 1).float().mean()), 4)]


res["table"] = {n: dict(all=acc(r), **{f"S{k + 1}": acc(r[te_sty == k]) for k in range(5)}) for n, r in A.items()}
res["M_mean_per_layer"] = {l: acc(PT["M_mean"][l]) for l in READ}
# word-level paired bootstrap
uw = sorted(set(wi[TE].tolist()))
pos_w = {w: j for j, w in enumerate(uw)}
wid = torch.tensor([pos_w[w] for w in wi[TE].tolist()])


def boot(x, y, k=10, n=2000):
    dd = (x <= k).float() - (y <= k).float()
    per = torch.zeros(len(uw)).index_add_(0, wid, dd)
    cnt = torch.zeros(len(uw)).index_add_(0, wid, torch.ones(len(dd)))
    gen = torch.Generator().manual_seed(0)
    out = sorted(float(per[s].sum() / cnt[s].sum()) for s in (torch.randint(len(uw), (len(uw),), generator=gen)
                                                              for _ in range(n)))
    return [round(float(dd.mean()), 4), round(out[int(.025 * n)], 4), round(out[int(.975 * n)], 4)]


res["boot"] = {}
for a, b in (("M_mean", "bagA"), ("M_mean", "bagB"), ("mix Jgen(mean)+M_mean", "mix Jgen(mean)+bagA"),
             ("mix Jgen(mean)+M_mean", "mix Jgen(mean)+bagB"), ("M_mean L4 only", "bagA"), ("M_mean L4 only", "bagB"),
             ("M_mean L12 only", "bagA"), ("M_mean L12 only", "bagB"), ("mix Jgen(mean)+M_mean", "ask the model"),
             ("mix Jgen(mean)+bagB", "Jgen mean")):
    res["boot"][f"{a} - {b}"] = dict(top10=boot(A[a], A[b]), top1=boot(A[a], A[b], k=1))
# overlap best reader vs ask
best = A["mix Jgen(mean)+M_mean"] <= 10
ask = ASKR <= 10
res["overlap_top10"] = dict(both=int((best & ask).sum()), reader_only=int((best & ~ask).sum()),
                            ask_only=int((~best & ask).sum()), neither=int((~best & ~ask).sum()), n=len(ask))
wb = defaultdict(lambda: [False, False])
for j, w in enumerate(wi[TE].tolist()):
    wb[w][0] |= bool(best[j]); wb[w][1] |= bool(ask[j])
res["overlap_words_any_sentence"] = dict(both=sum(a and b for a, b in wb.values()),
                                         reader_only=sum(a and not b for a, b in wb.values()),
                                         ask_only=sum(b and not a for a, b in wb.values()),
                                         neither=sum(not a and not b for a, b in wb.values()), n=len(wb))
TEl = TE.tolist()
ex = [j for j in range(0, len(TEl), len(TEl) // 12)][:12]
res["ask_examples"] = [dict(word=words_s[TEl[j]], style=style[TEl[j]] + 1, sent=sents[TEl[j]], ask_top3=F["ask_top"][j],
                            ask_rank=int(ASKR[j]), reader_rank=int(A["mix Jgen(mean)+M_mean"][j])) for j in ex]
json.dump(res, open(f"{OUT}/ctrl_results.json", "w"), indent=1, ensure_ascii=False)
log(f"done {time.time() - t00:.0f}s")
print("CTRL_DONE")
