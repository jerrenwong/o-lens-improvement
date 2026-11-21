"""dq1: use the step-1 labels without shrinking the data. Same M3 recipe as fit2.py / scripts/al2/fit.py ((eps, c) per
layer and variant chosen on ALL validation sentences), generalised to per-sentence class labels and weights:
  class mean = weighted mean of the class's sentences; within-class covariance Sw = sum_i w_i r_i r_i^T / sum_i w_i
  (with all w = 1 this is exactly class_stats of fit.py; the recipe uses no other per-class count).
Categories: rules of scripts/dq1/size_relabel.py (copied; the relabel list is checked against .hf27/sub2/dq1_sizing.json).
Variants (train-word sentences only):
  V0 ALL; V1_k CLEAN-MEAN (k = 1, 2, 4): a word's sentences are its inferable ones (rank <= 10) if it has >= k, else all;
  V2_w WEIGHTED (w = .25, .5): weight 1 inferable, w others;
  V3 RELABEL: SUB, GIVE, same-word-other-form (original label) + relabel pool under the model's label;
     other three NOT categories dropped; relabels removed if the new label equals (case-insensitive) or shares a
     5-letter stem (size_relabel.stem) with a validation/test word, or has no single-token form;
  V4 RELABEL-KEEP: all train sentences, relabel pool under the model's label (removed relabels dropped);
  V5 BOTH: ALL (original labels) + a second copy of the kept relabel sentences under the model's label;
  V6 V4 with the kept new labels permuted among those sentences (seed 0).
New label classes: a label equal (case-insensitive) to a train word joins that word's class; otherwise a new class with
target built as scripts/al2/collect.py builds meta["T"]. Writes .hf27/sub2/dq1/fit_ranks.pt, relabel.json,
fit_choice.json, maps .hf27/sub2/M_dq1_<V>.pt (e2 format). Usage: dq1_fit.py [--layers ..] [--tiny K] [--out DIR]"""
import argparse, collections, json, os, random, re, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
_argv, sys.argv = sys.argv, sys.argv[:1]  # al2_common imports g1_gen, which parses sys.argv
from al2_common import R, BIG, BIG1, log, free_mb  # noqa: E402
sys.argv = _argv
from workspace_lens.utils import load_lens_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="4,8,12,16,20,24,28")
ap.add_argument("--tiny", type=int, default=0)
ap.add_argument("--out", default=f"{R}/.hf27/sub2/dq1")
ap.add_argument("--need", type=int, default=12000)
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"
OUT = args.out
os.makedirs(OUT, exist_ok=True)
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [int(x) for x in args.layers.split(",")]
LAMS = [0.1, 0.3, 1.0]
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style, role, sents = meta["words"], meta["split"], meta["style"], meta["role"], meta["sents"]
N = len(words_s)
LB = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt")
rank, rs, top3, top3s = LB["rank"], LB["rank_shuf"], LB["top3"], LB["top3_shuf"]
strs = torch.load(f"{BIG1}/meta.pt", weights_only=False)["strs"]


# ---------------- categories: copied from scripts/dq1/size_relabel.py ----------------
def tk(x): return [strs[int(i)].strip() for i in x]


def stem(a, b):
    a, b = a.lower(), b.lower(); m = min(len(a), len(b)); k = max(4, min(5, m)); return m >= 4 and a[:k] == b[:k]


cat, relab = [], {}
for i in range(N):
    r, q = int(rank[i]), int(rs[i]); t = tk(top3[i]); ts = tk(top3s[i])
    toks = set(re.findall(r"[a-z']+", sents[i].lower()))
    if r <= 10: c = "GIVE" if q <= 10 else "SUB"
    else:
        real = [x for x in t if x and x[0].isalpha() and len(x) >= 3]
        if not real: c = "NOT: no word answer"
        elif any(stem(x, words_s[i]) for x in real): c = "NOT: same word, other form"
        elif real[0].lower() in toks or any(stem(real[0], y) for y in toks): c = "NOT: copies a visible word"
        elif real[0].lower() in [y.lower() for y in ts]: c = "NOT: other word, also from shuffled"
        else:
            c = "NOT: other unstated word, needs order"; relab[i] = real[0].lower()
    cat.append(c)
# -------------------------------------------------------------------------------------
ref = json.load(open(f"{R}/.hf27/sub2/dq1_sizing.json"))
assert {int(i): x for i, x in ref["relabel"]} == relab, "relabel list differs from dq1_sizing.json"
assert collections.Counter(cat) == collections.Counter(ref["cat"]), "categories differ"
log(f"categories match dq1_sizing.json: {dict(collections.Counter(cat))}")

# ---------------- vocab / targets ----------------
U = torch.load(f"{BIG1}/U.pt")
g_cpu = meta["g"].float()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def target(w):  # as scripts/al2/collect.py
    forms = {" " + w, w, " " + w.lower(), w.lower(), " " + w.lower().capitalize(), w.lower().capitalize()}
    ids = sorted({t[0] for t in tok(sorted(forms), add_special_tokens=False).input_ids if len(t) == 1})
    if not ids:
        return None
    u = U[torch.tensor(ids)].float()
    return (u / u.norm(dim=-1, keepdim=True)).mean(0) / g_cpu


chkw = random.Random(0).sample(sorted(role), 50)
tdiff = max(float((target(w) - meta["T"][w].float()).abs().max() / meta["T"][w].float().abs().max()) for w in chkw)
log(f"target rebuild check on 50 original words: max rel diff {tdiff:.2e}")
assert tdiff < 1e-3, tdiff

W_all = sorted(role)
widx = {w: i for i, w in enumerate(W_all)}
valtest = [w for w in W_all if role[w] != "train"]
vt_lower = {w.lower() for w in valtest}
train_lower = defaultdict(list)
for w in W_all:
    if role[w] == "train":
        train_lower[w.lower()].append(w)
TRN_all = [i for i in range(N) if split[i] == "train"]
pool = [i for i in TRN_all if i in relab]
labels = sorted({relab[i] for i in pool})
why = {}
new_cls, new_T = {}, []
for x in labels:
    if x in vt_lower:
        why[x] = "equals val/test word"; continue
    if any(stem(x, v) for v in valtest):
        why[x] = "stem shared with val/test word"; continue
    if x in train_lower:
        new_cls[x] = widx[sorted(train_lower[x])[0]]; why[x] = "joins train word class"; continue
    t = target(x)
    if t is None:
        why[x] = "no single-token form"; continue
    new_cls[x] = len(W_all) + len(new_T); new_T.append(t); why[x] = "new class"
kept = [i for i in pool if relab[i] in new_cls]
removed = [i for i in pool if relab[i] not in new_cls]
rstat = dict(pool_train_sentences=len(pool), pool_all_sentences=len(relab), distinct_labels=len(labels),
             removed_sentences=len(removed), kept_sentences=len(kept),
             removed_by_reason={k: sum(why[relab[i]] == k for i in removed) for k in set(why.values())},
             kept_by_reason={k: sum(why[relab[i]] == k for i in kept) for k in set(why.values())},
             labels_by_reason=dict(collections.Counter(why.values())),
             distinct_labels_kept=len({relab[i] for i in kept}), new_classes=len(new_T),
             label_equals_original_word=sum(relab[i] == words_s[i].lower() for i in kept))
ex = random.Random(1).sample(kept, 20)
rstat["examples"] = [dict(i=i, old=words_s[i], new=relab[i], style=style[i] + 1, sent=sents[i]) for i in ex]
log("relabel " + json.dumps({k: v for k, v in rstat.items() if k != "examples"}))
json.dump(rstat, open(f"{OUT}/relabel.json", "w"), indent=1)

# ---------------- variants: (row idx, class id, weight) ----------------
inf = (rank <= 10)
wi_cpu = torch.tensor([widx[w] for w in words_s])
TRN = torch.tensor(TRN_all)
if args.tiny:
    keepw = set(sorted(set(wi_cpu[TRN].tolist()))[: args.tiny])
    TRN = torch.tensor([i for i in TRN_all if int(wi_cpu[i]) in keepw])
    kept = [i for i in kept if int(wi_cpu[i]) in keepw]
    removed = [i for i in removed if int(wi_cpu[i]) in keepw]
trset = set(TRN.tolist())
kept_t = torch.tensor(kept, dtype=torch.long)
kept_lab = torch.tensor([new_cls[relab[i]] for i in kept], dtype=torch.long)
rm_set = set(removed)
VAR = {}
VAR["V0"] = (TRN, wi_cpu[TRN], torch.ones(len(TRN)))
ninf = collections.Counter(int(wi_cpu[i]) for i in TRN.tolist() if inf[i])
for k in (1, 2, 4):
    idx = torch.tensor([i for i in TRN.tolist() if (inf[i] or ninf[int(wi_cpu[i])] < k)])
    VAR[f"V1_k{k}"] = (idx, wi_cpu[idx], torch.ones(len(idx)))
for w in (0.25, 0.5):
    VAR[f"V2_w{w}"] = (TRN, wi_cpu[TRN], torch.where(inf[TRN], 1.0, w))
keep_orig = {"SUB", "GIVE", "NOT: same word, other form"}
a = torch.tensor([i for i in TRN.tolist() if cat[i] in keep_orig])
VAR["V3"] = (torch.cat([a, kept_t]), torch.cat([wi_cpu[a], kept_lab]), torch.ones(len(a) + len(kept)))
b = torch.tensor([i for i in TRN.tolist() if i not in relab])  # non-pool train sentences
VAR["V4"] = (torch.cat([b, kept_t]), torch.cat([wi_cpu[b], kept_lab]), torch.ones(len(b) + len(kept)))
VAR["V5"] = (torch.cat([TRN, kept_t]), torch.cat([wi_cpu[TRN], kept_lab]), torch.ones(len(TRN) + len(kept)))
perm = torch.randperm(len(kept), generator=torch.Generator().manual_seed(0))
VAR["V6"] = (torch.cat([b, kept_t]), torch.cat([wi_cpu[b], kept_lab[perm]]), torch.ones(len(b) + len(kept)))
vinfo = {k: dict(n_rows=len(v[0]), n_classes=len(set(v[1].tolist())), n_distinct_sentences=len(set(v[0].tolist())),
                 weight_sum=round(float(v[2].sum()), 1)) for k, v in VAR.items()}
log("variants " + json.dumps(vinfo))

# ---------------- GPU ----------------
while free_mb() < args.need:
    log(f"waiting for GPU: free {free_mb()} MiB < {args.need}")
    time.sleep(60)
U = U.to(dev)
V, d = U.shape
g = meta["g"].to(dev)
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
by_norm = defaultdict(list)
for i, s in enumerate(strs):
    by_norm[s.strip().lower()].append(i)
forms = [by_norm[w.lower()] for w in W_all]
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
TW = torch.stack([meta["T"][w] for w in W_all] + new_T).to(dev).float()
wi = wi_cpu.to(dev)
spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("val", "test")}
VAL = spl["val"].nonzero()[:, 0]
TE = spl["test"].nonzero()[:, 0]
if args.tiny:
    VAL, TE = VAL[:300], TE[:300]
VARD = {k: (a_.to(dev), l_.to(dev), w_.to(dev).float()) for k, (a_, l_, w_) in VAR.items()}


def ranks_of(vecs, wid, bs=256):
    out = []
    for i in range(0, vecs.shape[0], bs):
        lg = ((vecs[i:i + bs].float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid[i:i + bs]]).max(1).values
        out.append((lg > best[:, None]).sum(1) + 1)
        del lg
    return torch.cat(out)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class Ridge:  # fit.py
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


def class_stats_w(X, idx, lab, wt):
    uw, inv = torch.unique(lab, return_inverse=True)
    Mn = torch.zeros(len(uw), d, device=dev)
    cw = torch.zeros(len(uw), device=dev).index_add_(0, inv, wt)
    for i in range(0, len(idx), 16384):
        Mn.index_add_(0, inv[i:i + 16384], X[idx[i:i + 16384]] * wt[i:i + 16384, None])
    Mn /= cw[:, None]
    Sw = torch.zeros(d, d, device=dev, dtype=torch.float64)
    for i in range(0, len(idx), 16384):
        Rw = X[idx[i:i + 16384]] - Mn[inv[i:i + 16384]]
        Sw += ((Rw * wt[i:i + 16384, None]).T @ Rw).double()
    return uw, Mn, Sw / wt.sum()


def whitener(Sw, f):  # fit.py
    s, Q = torch.linalg.eigh(Sw)
    s = s.clamp_min(0)
    return (Q @ torch.diag((s + f * s.mean()) ** -0.5) @ Q.T).float()


def fit_M3(X, var, epsf=EPSF, grid=GRID):
    uw, Mn, Sw = class_stats_w(X, *var)
    Tm = TW[uw]
    for f in epsf:
        P = whitener(Sw, f)
        r = Ridge(Mn @ P); r.setY(Tm)
        for c in grid:
            yield dict(f=f, c=c), (Mn.mean(0), P @ r.W(c), r.my)
        del r, P


def apply(m, x):
    mx, W, my = m
    return (x - mx) @ W + my


def choose(X, var, val_idx):
    best = None
    for cfg, m in fit_M3(X, var):
        r = ranks_of(apply(m, X[val_idx]), wi[val_idx])
        z = (round(float((r <= 10).float().mean()), 6), float((r <= 1).float().mean()))
        if best is None or z > best[0]:
            best = (z, cfg, m)
    return best


def pack(m):
    mx, W, my = m
    sc = W.abs().max().item()
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.float().cpu())


JPP = {int(l): v.float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}
VR, TR = defaultdict(dict), defaultdict(dict)
choice = defaultdict(dict)
maps_save = defaultdict(dict)


def do_layer(l):
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    J = JPP[l].to(dev)
    jv, jt = X[VAL] @ J.T, X[TE] @ J.T
    VR["J++"][l] = ranks_of(jv, wi[VAL]).cpu()
    TR["J++"][l] = ranks_of(jt, wi[TE]).cpu()
    for name, var in VARD.items():
        z, cfg, m = choose(X, var, VAL)
        choice[name][l] = dict(cfg=cfg, val_top10=z[0])
        mv, mt = apply(m, X[VAL]), apply(m, X[TE])
        VR[f"M|{name}"][l] = ranks_of(mv, wi[VAL]).cpu()
        TR[f"M|{name}"][l] = ranks_of(mt, wi[TE]).cpu()
        for lam in LAMS:
            VR[f"mix{lam}|{name}"][l] = ranks_of(unit(jv) + lam * unit(mv), wi[VAL]).cpu()
            TR[f"mix{lam}|{name}"][l] = ranks_of(unit(jt) + lam * unit(mt), wi[TE]).cpu()
        maps_save[name][l] = pack(m)
        del m, mv, mt
        torch.cuda.empty_cache()


for l in READ:
    t0 = time.time()
    while True:
        try:
            do_layer(l)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"CUDA OOM at L{l}; free {free_mb()} MiB; retry in 60 s")
            time.sleep(60)
    torch.cuda.empty_cache()
    log(f"L{l} ({time.time() - t0:.0f}s) test top10 " +
        " ".join(f"{n} {float((TR[n][l] <= 10).float().mean()):.3f}" for n in ["J++"] + [f"M|{k}" for k in VARD]) +
        " | cfg " + " ".join(f"{k}:{choice[k][l]['cfg']['f']},{choice[k][l]['cfg']['c']}" for k in VARD))

torch.save(dict(VR=dict(VR), TR=dict(TR), val_rows=VAL.cpu(), te_rows=TE.cpu(), layers=READ, cat=cat,
                variants={k: v[0] for k, v in VAR.items()}), f"{OUT}/fit_ranks.pt")
json.dump(dict(choice={k: {str(l): c for l, c in v.items()} for k, v in choice.items()}, variants=vinfo, layers=READ),
          open(f"{OUT}/fit_choice.json", "w"), indent=1)
if READ == [4, 8, 12, 16, 20, 24, 28] and not args.tiny:
    for name in VARD:
        torch.save(maps_save[name], f"{R}/.hf27/sub2/M_dq1_{name}.pt")
log(f"done {time.time() - t00:.0f}s")
print("DQ1_FIT_DONE")
