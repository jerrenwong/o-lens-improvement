"""al2 step 3: al1's M3 ("noise-aware" ridge: whiten by within-word covariance, ridge on per-word mean activations)
fitted on (a) all five styles M_all, (b) all but S3 M_noS3, (c) S1+S2 M_12; shuffled-label control for M_all;
learning curve for M_all. M is a LEARNED linear map, not a Jacobian. Per layer, (eps, c) chosen on validation words,
using only validation sentences of the styles the map was trained on. Mix unit(J++ h) + lam unit(M h): lam in
{0.1, 0.3, 1.0} chosen on validation words (any-of-7 top-10, same style restriction). Ranking: (v*g) @ W_U^T over the
full vocabulary, benchmark punctuation filter (no letter/digit -> excluded); hit = any surface form of the word.
Writes {BIG}/fit_results.json, M_all.pt, M_shuf.pt (e2 map format), ranks.pt."""
import json, os, random, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al2_common import R, BIG, BIG1, log
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [4, 8, 12, 16, 20, 24, 28]
LAMS = [0.1, 0.3, 1.0]
SIZES = [500, 1000, 2000, 4000]
VARS = {"M_all": {0, 1, 2, 3, 4}, "M_noS3": {0, 1, 3, 4}, "M_12": {0, 1}}
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style = meta["words"], meta["split"], meta["style"]
role = meta["role"]
g = meta["g"].to(dev)
U = torch.load(f"{BIG1}/U.pt").to(dev)
V, d = U.shape
strs = json.load(open(f"{BIG}/strs.json")) if os.path.exists(f"{BIG}/strs.json") else torch.load(
    f"{BIG1}/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
by_norm = defaultdict(list)
for i, s in enumerate(strs):
    by_norm[s.strip().lower()].append(i)
W_all = sorted(role)
widx = {w: i for i, w in enumerate(W_all)}
forms = [by_norm[w.lower()] for w in W_all]
assert all(forms), [w for w, f in zip(W_all, forms) if not f][:10]
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
TW = torch.stack([meta["T"][w] for w in W_all]).to(dev).float()
wi = torch.tensor([widx[w] for w in words_s], device=dev)
sty = torch.tensor(style, device=dev)
spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("train", "val", "test")}


def rows(r, styles):
    m = spl[r] & torch.isin(sty, torch.tensor(sorted(styles), device=dev))
    return m.nonzero()[:, 0]


VAL = {v: rows("val", s) for v, s in VARS.items()}
TE = rows("test", {0, 1, 2, 3, 4})
te_sty = sty[TE].cpu()
log(f"sentences {len(words_s)}; train {int(spl['train'].sum())} val {int(spl['val'].sum())} test {len(TE)}; "
    f"words {len(W_all)}; vocab {V}; excluded {int(excl.sum())}; max forms {mf}")


def ranks_of(vecs, wid, bs=512):
    out = []
    for i in range(0, vecs.shape[0], bs):
        lg = ((vecs[i:i + bs].float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid[i:i + bs]]).max(1).values
        out.append((lg > best[:, None]).sum(1) + 1)
    return torch.cat(out)


def topk_tokens(vecs, k=8):
    lg = ((vecs.float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
    return lg.topk(k, -1).indices


def acc(r):
    return dict(top10=round(float((r <= 10).float().mean()), 4), top1=round(float((r <= 1).float().mean()), 4),
                n=int(len(r)))


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


JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}
train_all = rows("train", {0, 1, 2, 3, 4})
tr_words = sorted(set(wi[train_all].tolist()))
tw = list(tr_words); random.Random(0).shuffle(tw)
pw = tw[:]; random.Random(1).shuffle(pw)
PM = torch.arange(len(W_all), device=dev)
PM[torch.tensor(tw, device=dev)] = torch.tensor(pw, device=dev)
VR = defaultdict(dict)   # VR[name][l] = val ranks (rows VAL[var])
TR = defaultdict(dict)   # TR[name][l] = test ranks (TE rows)
res = dict(choice=defaultdict(dict), n_train_words=len(tr_words))
maps_save = {}
for l in READ:
    t0 = time.time()
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    jv = {v: X[VAL[v]] @ JPP[l].T for v in VARS}
    jt = X[TE] @ JPP[l].T
    TR["J++"][l] = ranks_of(jt, wi[TE]).cpu()
    for v, ss in VARS.items():
        VR[f"J++|{v}"][l] = ranks_of(jv[v], wi[VAL[v]]).cpu()
        z, cfg, m = choose(X, rows("train", ss), VAL[v])
        res["choice"][v][l] = dict(cfg=cfg, val_top10=z[0])
        mv, mt = apply(m, X[VAL[v]]), apply(m, X[TE])
        VR[v][l] = ranks_of(mv, wi[VAL[v]]).cpu()
        TR[v][l] = ranks_of(mt, wi[TE]).cpu()
        for lam in LAMS:
            VR[f"mix{lam}|{v}"][l] = ranks_of(unit(jv[v]) + lam * unit(mv), wi[VAL[v]]).cpu()
            TR[f"mix{lam}|{v}"][l] = ranks_of(unit(jt) + lam * unit(mt), wi[TE]).cpu()
        if v == "M_all":
            maps_save[("M_all", l)] = tuple(t.cpu() for t in m)
            # shuffled-label control (same cfg)
            ms = next(iter(fit_M3(X, train_all, lambda ids: TW[PM[ids]], epsf=[cfg["f"]], grid=[cfg["c"]])))[1]
            maps_save[("M_shuf", l)] = tuple(t.cpu() for t in ms)
            mst = apply(ms, X[TE])
            TR["M_shuf"][l] = ranks_of(mst, wi[TE]).cpu()
            for lam in LAMS:
                TR[f"mix{lam}|M_shuf"][l] = ranks_of(unit(jt) + lam * unit(mst), wi[TE]).cpu()
            # learning curve: eps fixed at chosen, c re-chosen on val per size
            for n in SIZES + ["all"]:
                if n == "all":
                    lm = m
                else:
                    keep = torch.tensor(sorted(tw[:n]), device=dev)
                    tr = train_all[torch.isin(wi[train_all], keep)]
                    _, _, lm = choose(X, tr, VAL[v], epsf=[cfg["f"]])
                lv, lt = apply(lm, X[VAL[v]]), apply(lm, X[TE])
                VR[f"lc{n}"][l] = ranks_of(lv, wi[VAL[v]]).cpu()
                TR[f"lc{n}"][l] = ranks_of(lt, wi[TE]).cpu()
                for lam in LAMS:
                    VR[f"lcmix{lam}|{n}"][l] = ranks_of(unit(jv[v]) + lam * unit(lv), wi[VAL[v]]).cpu()
                    TR[f"lcmix{lam}|{n}"][l] = ranks_of(unit(jt) + lam * unit(lt), wi[TE]).cpu()
                del lm
        del m
    log(f"L{l} ({time.time() - t0:.0f}s) " + " ".join(f"{v}:{res['choice'][v][l]}" for v in VARS) +
        " | test top10 " + " ".join(f"{n} {float((TR[n][l] <= 10).float().mean()):.3f}" for n in
                                    ["J++", "M_all", "M_noS3", "M_12", "M_shuf"]))
    del X
    torch.cuda.empty_cache()


def any7(store):
    return torch.stack([store[l] for l in READ]).min(0).values


def t10(r):
    return float((r <= 10).float().mean())


lam_of = {}
for v in VARS:
    vm = {lam: t10(any7(VR[f"mix{lam}|{v}"])) for lam in LAMS}
    lam_of[v] = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    res[f"mixval_{v}"] = vm
    for l in READ:
        TR[f"mix|{v}"][l] = TR[f"mix{lam_of[v]}|{v}"][l]
for l in READ:
    TR["mix|M_shuf"][l] = TR[f"mix{lam_of['M_all']}|M_shuf"][l]
lc_lam = {}
for n in SIZES + ["all"]:
    vm = {lam: t10(any7(VR[f"lcmix{lam}|{n}"])) for lam in LAMS}
    lc_lam[n] = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
res["lam"] = lam_of
res["lc_lam"] = lc_lam
res["val_any7"] = {k: t10(any7(VR[k])) for k in ["J++|M_all", "M_all", "M_noS3", "M_12"]}
log(f"lam chosen {lam_of}; lc lam {lc_lam}; val any7 {res['val_any7']}")

NAMES = ["J++", "M_all", "M_noS3", "M_12", "M_shuf", "mix|M_all", "mix|M_noS3", "mix|M_12", "mix|M_shuf"]
res["T1_layers"] = {l: {n: acc(TR[n][l]) for n in NAMES} for l in READ}
res["T1_any7"] = {n: acc(any7(TR[n])) for n in NAMES}
res["by_style_any7"] = {n: {f"S{k + 1}": acc(any7(TR[n])[te_sty == k]) for k in range(5)} for n in NAMES}
res["by_style_L28"] = {n: {f"S{k + 1}": acc(TR[n][28][te_sty == k]) for k in range(5)} for n in NAMES}
res["T2_lc"] = {str(n): dict(M=acc(any7(TR[f"lc{n}"])), mix=acc(any7(TR[f"lcmix{lc_lam[n]}|{n}"])), lam=lc_lam[n])
                for n in SIZES + ["all"]}
s3 = te_sty == 2
res["T3_S3"] = {n: dict(any7=acc(any7(TR[n])[s3]), per_layer={l: acc(TR[n][l][s3]) for l in READ}) for n in NAMES}
# complementarity on test (any7)
a, b = any7(TR["J++"]) <= 10, any7(TR["M_all"]) <= 10
res["complement"] = dict(both=int((a & b).sum()), jpp_only=int((a & ~b).sum()), M_only=int((~a & b).sum()),
                         neither=int((~a & ~b).sum()))
# examples (L28)
Xl = torch.load(f"{BIG}/X_L28.pt")
ex_rows = random.Random(5).sample(TE.tolist(), 15)
pos = {i: j for j, i in enumerate(TE.tolist())}
xm = Xl[ex_rows].to(dev).float()
tm = topk_tokens(apply(tuple(t.to(dev) for t in maps_save[("M_all", 28)]), xm))
tj = topk_tokens(xm @ JPP[28].T)
res["examples_L28"] = [dict(word=words_s[i], style=style[i] + 1, sent=meta["sents"][i],
                            M=[strs[t].strip() for t in tm[j].tolist()], M_rank=int(TR["M_all"][28][pos[i]]),
                            J=[strs[t].strip() for t in tj[j].tolist()], J_rank=int(TR["J++"][28][pos[i]]))
                       for j, i in enumerate(ex_rows)]


def pack(m):
    mx, W, my = m
    sc = W.abs().max().item()
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.float().cpu())


torch.save({l: pack(maps_save[("M_all", l)]) for l in READ}, f"{BIG}/M_all.pt")
torch.save({l: pack(maps_save[("M_shuf", l)]) for l in READ}, f"{BIG}/M_shuf.pt")
torch.save(dict(TR=dict(TR), te_rows=TE.cpu(), te_style=te_sty), f"{BIG}/ranks.pt")
res["choice"] = {v: {str(l): c for l, c in ch.items()} for v, ch in res["choice"].items()}
json.dump(res, open(f"{BIG}/fit_results.json", "w"), indent=1, default=str)
log(f"done {time.time() - t00:.0f}s")
print("FIT_DONE")
