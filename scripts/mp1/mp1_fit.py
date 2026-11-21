"""mp1 step 2: readers on (a) token mean, (b) last-half mean, (c) last token (al2 X_L*), test words of al2.
J++ released / generated-fit on each input; al2 M3 map refitted on each input (M_mean, M_half, M_all) with (eps, c)
chosen on validation words (all styles, as al2 M_all); mixes unit(Jgen h) + lam unit(M h), lam in {0.1,0.3,1.0}
chosen on validation (any-of-7 top-10); cross mix unit(Jgen h_last) + lam unit(M_mean h_mean); shuffled-label M_mean.
Per-token J++ (i)/(ii) ranks come from mp1_collect.py phase 2.  M is a LEARNED linear map, not a Jacobian.
Writes .hf27/mp1/fit_results.json, M_mean.pt (e2 map format), ranks.pt."""
import json, os, random, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
from al2_common import R, BIG, BIG1, log
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
OUT = f"{R}/.hf27/mp1"
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [4, 8, 12, 16, 20, 24, 28]
LAMS = [0.1, 0.3, 1.0]
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style, role = meta["words"], meta["split"], meta["style"], meta["role"]
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
TRN = spl["train"].nonzero()[:, 0]
VAL = spl["val"].nonzero()[:, 0]
TE = spl["test"].nonzero()[:, 0]
te_sty = sty[TE].cpu()


def ranks_of(vecs, wid, bs=512):
    out = []
    for i in range(0, vecs.shape[0], bs):
        lg = ((vecs[i:i + bs].float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid[i:i + bs]]).max(1).values
        out.append((lg > best[:, None]).sum(1) + 1)
    return torch.cat(out)


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


def lensd(p):
    return {int(l): v.to(dev).float() for l, v in
            load_lens_file(f"{R}/{p}", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}


JL = {"Jrel": lensd("artifacts/baselines/jpp9b_prefit.pt"), "Jgen": lensd("runs/w1/jpp_Afull_Wtrain.pt")}
INP = {"last": lambda l: f"{BIG}/X_L{l}.pt", "mean": lambda l: f"{OUT}/Xmean_L{l}.pt",
       "half": lambda l: f"{OUT}/Xhalf_L{l}.pt"}
MNAME = {"last": "M_all", "mean": "M_mean", "half": "M_half"}
tw = sorted(set(wi[TRN].tolist())); random.Random(0).shuffle(tw)
pw = tw[:]; random.Random(1).shuffle(pw)
PM = torch.arange(len(W_all), device=dev)
PM[torch.tensor(tw, device=dev)] = torch.tensor(pw, device=dev)
VR = defaultdict(dict)
TR = defaultdict(dict)
res = dict(choice=defaultdict(dict))
maps_save = {}
for l in READ:
    t0 = time.time()
    jv, jt, mv, mt = {}, {}, {}, {}
    for x, fn in INP.items():
        X = torch.load(fn(l)).to(dev).float()
        for jn, J in JL.items():
            jv[(jn, x)] = X[VAL] @ J[l].T
            jt[(jn, x)] = X[TE] @ J[l].T
            VR[f"{jn}|{x}"][l] = ranks_of(jv[(jn, x)], wi[VAL]).cpu()
            TR[f"{jn}|{x}"][l] = ranks_of(jt[(jn, x)], wi[TE]).cpu()
        z, cfg, m = choose(X, TRN, VAL)
        res["choice"][MNAME[x]][l] = dict(cfg=cfg, val_top10=z[0])
        mv[x], mt[x] = apply(m, X[VAL]), apply(m, X[TE])
        VR[MNAME[x]][l] = ranks_of(mv[x], wi[VAL]).cpu()
        TR[MNAME[x]][l] = ranks_of(mt[x], wi[TE]).cpu()
        if x == "mean":
            maps_save[l] = tuple(t.cpu() for t in m)
            ms = next(iter(fit_M3(X, TRN, lambda ids: TW[PM[ids]], epsf=[cfg["f"]], grid=[cfg["c"]])))[1]
            mst = apply(ms, X[TE])
            TR["M_mean_shuf"][l] = ranks_of(mst, wi[TE]).cpu()
            for lam in LAMS:
                TR[f"mix{lam}|Jgen+M_mean_shuf (mean)"][l] = ranks_of(unit(jt[("Jgen", "mean")]) + lam * unit(mst), wi[TE]).cpu()
            del ms, mst
        del m, X
        torch.cuda.empty_cache()
    for lam in LAMS:
        for x in INP:
            VR[f"mix{lam}|Jgen+{MNAME[x]} ({x})"][l] = ranks_of(unit(jv[("Jgen", x)]) + lam * unit(mv[x]), wi[VAL]).cpu()
            TR[f"mix{lam}|Jgen+{MNAME[x]} ({x})"][l] = ranks_of(unit(jt[("Jgen", x)]) + lam * unit(mt[x]), wi[TE]).cpu()
        VR[f"mix{lam}|Jgen(last)+M_mean(mean)"][l] = ranks_of(unit(jv[("Jgen", "last")]) + lam * unit(mv["mean"]), wi[VAL]).cpu()
        TR[f"mix{lam}|Jgen(last)+M_mean(mean)"][l] = ranks_of(unit(jt[("Jgen", "last")]) + lam * unit(mt["mean"]), wi[TE]).cpu()
    log(f"L{l} ({time.time() - t0:.0f}s) " + " ".join(f"{k}:{v[l]}" for k, v in res["choice"].items()) +
        " | test top10 " + " ".join(f"{n} {float((TR[n][l] <= 10).float().mean()):.3f}" for n in
                                    ["Jrel|last", "Jrel|mean", "Jgen|last", "Jgen|mean", "Jgen|half",
                                     "M_all", "M_mean", "M_half", "M_mean_shuf"]))
    del jv, jt, mv, mt
    torch.cuda.empty_cache()


def any7(store):
    return torch.stack([store[l] for l in READ]).min(0).values


def t10(r):
    return float((r <= 10).float().mean())


MIXES = [f"Jgen+{MNAME[x]} ({x})" for x in INP] + ["Jgen(last)+M_mean(mean)"]
res["lam"], res["mixval"] = {}, {}
for mx_ in MIXES:
    vm = {lam: t10(any7(VR[f"mix{lam}|{mx_}"])) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    res["lam"][mx_], res["mixval"][mx_] = lam, vm
    for l in READ:
        TR[f"mix|{mx_}"][l] = TR[f"mix{lam}|{mx_}"][l]
for l in READ:
    TR["mix|Jgen+M_mean_shuf (mean)"][l] = TR[f"mix{res['lam']['Jgen+M_mean (mean)']}|Jgen+M_mean_shuf (mean)"][l]
res["val_any7"] = {k: t10(any7(VR[k])) for k in VR if not k.startswith("mix")}

# per-token readers (phase 2 of mp1_collect.py)
PT = torch.load(f"{OUT}/pertok_ranks.pt")
assert torch.equal(PT["te_rows"], TE.cpu())
for (ln, k, l), r in PT["RK"].items():
    TR[f"{ln}|tok_{k}"][l] = r
res["winpos_mean"] = {f"{ln}|L{l}": float(v.mean()) for (ln, l), v in PT["WINPOS"].items()}
# sanity: (i) raw mean of per-token logits == J on the token mean
res["check_i_raw_vs_mean"] = {f"{ln}|L{l}": float((TR[f"{ln}|tok_i_raw"][l] == TR[f"{ln}|mean"][l]).float().mean())
                              for ln in JL for l in READ}


def acc(r):
    return [round(float((r <= 10).float().mean()), 4), round(float((r <= 1).float().mean()), 4)]


NAMES = [n for n in TR if not n.startswith("mix0") and not n.startswith("mix1")]
res["T_layers"] = {n: {l: acc(TR[n][l]) for l in READ} for n in NAMES}
res["T_any7"] = {n: acc(any7(TR[n])) for n in NAMES}
res["T_style_any7"] = {n: {f"S{k + 1}": acc(any7(TR[n])[te_sty == k]) for k in range(5)} for n in NAMES}
res["n_test"] = len(TE)


def pack(m):
    mx, W, my = m
    sc = W.abs().max().item()
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.float().cpu())


torch.save({l: pack(maps_save[l]) for l in READ}, f"{OUT}/M_mean.pt")
torch.save(dict(TR=dict(TR), te_rows=TE.cpu(), te_style=te_sty), f"{OUT}/ranks.pt")
res["choice"] = {v: {str(l): c for l, c in ch.items()} for v, ch in res["choice"].items()}
json.dump(res, open(f"{OUT}/fit_results.json", "w"), indent=1, default=str)
log(f"done {time.time() - t00:.0f}s")
print("FIT_DONE")
