"""al3 step 4: fit and evaluate translation maps (al1/al2 "noise-aware" ridge M3, code copied from scripts/al2/fit.py:
whiten by within-word covariance, ridge on per-word mean last-token activations -> the word's mean unit unembedding
row / g; (eps, c) per layer chosen on validation words by top-10 (ties top-1)). M is a LEARNED map, not a Jacobian.
Maps:  a = al2 ALL (al2 train rows; chosen on al2 val words: reproduces al2 M_all)
       b = al3 only (al3 train rows; chosen on al3 val words)
       c = al2 + al3 (al2 train rows + new al3 train rows; al3 val)
       d = al3 inferable-only (al3 train rows with rank <= 10; al3 val)
       e = al2 ALL + al3 inferable-only (al2 train rows + new al3 train rows with rank <= 10; al3 val)
Readers: M alone; released J++ alone; mix unit(J++ h) + lam unit(M h), lam in {0.1, 0.3, 1.0} chosen per map on
validation words by any-of-7 top-10 (ties -> smaller lam): on al2 val words for T1, on al3 val words for T2/T3.
"best layer" = layer with the best val top-10 (same val set as lam). Ranking: full vocab, punctuation filter, any
surface form (al2 rule). Tests: T1 al2 test (all, nameable = keep in al2_words_judged), T2 al3 test (stem-separated;
by type, by label bin), T3 near. Paired word bootstrap (2,000) for c-a and b-a, any-of-7 top-10.
Usage: fit.py [--dev cuda|cpu] [--maps a,b,c,d,e] [--layers 4,...] [--tag NAME]"""
import argparse, json, os, random, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import R, BIG, BIG2, BIG1, RES, LAYERS, SNAME, log, free_mb  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cuda")
ap.add_argument("--maps", default="a,b,c,d,e")
ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = args.dev
if dev == "cpu":
    torch.set_num_threads(96)
else:
    torch.cuda.set_per_process_memory_fraction(0.42)
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [int(x) for x in args.layers.split(",")]
LAMS = [0.1, 0.3, 1.0]
MAPS = args.maps.split(",")
TYPES = ["PERSON", "PLACE", "THING", "EVENT", "STATE"]
t00 = time.time()

# ---------------- data ----------------
m2 = torch.load(f"{BIG2}/meta.pt", weights_only=False)
N2 = len(m2["words"])
role2 = m2["role"]
have3 = os.path.exists(f"{BIG}/meta.pt") and os.path.exists(f"{BIG}/X_L28.pt") and os.path.exists(f"{BIG}/labels_new.pt")
assert have3 or set(MAPS) <= {"a"}, "al3 data missing"
L2 = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt")
words_u, style_u = list(m2["words"]), list(m2["style"])
rank_u, rshuf_u = [int(x) for x in L2["rank"]], [int(x) for x in L2["rank_shuf"]]
if have3:
    m3 = torch.load(f"{BIG}/meta.pt", weights_only=False)
    L3 = torch.load(f"{BIG}/labels_new.pt")
    NN = len(m3["new_sents"])
    words_u += [None] * NN; style_u += [None] * NN
    rank_u += [int(x) for x in L3["rank"]]; rshuf_u += [int(x) for x in L3["rank_shuf"]]
    for r in m3["rows"]:
        if r["src"] == "new":
            words_u[N2 + r["new_idx"]] = r["word"]; style_u[N2 + r["new_idx"]] = r["style"]
    rows3 = m3["rows"]
    u3 = [r["al2_idx"] if r["src"] == "al2" else N2 + r["new_idx"] for r in rows3]
    concepts = {c["word"]: c for c in m3["concepts"]}
else:
    NN, rows3, u3, concepts = 0, [], [], {}
NU = N2 + NN
judged = {r["word"]: r for r in json.load(open(f"{R}/.hf27/sub1/al2_words_judged.json"))}
U = torch.load(f"{BIG1}/U.pt").to(dev)
if dev == "cpu":
    U = U.float()
V, d = U.shape
g = m2["g"].to(dev).float()
strs = json.load(open(f"{BIG2}/strs.json")) if os.path.exists(f"{BIG2}/strs.json") else torch.load(
    f"{BIG1}/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
by_norm = defaultdict(list)
for i, s in enumerate(strs):
    by_norm[s.strip().lower()].append(i)
W_all = sorted(set(role2) | set(concepts))
widx = {w: i for i, w in enumerate(W_all)}
forms = [by_norm[w.lower()] for w in W_all]
assert all(forms), [w for w, f in zip(W_all, forms) if not f][:10]
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
# targets: al2 meta T for al2 words; same recipe (al2 collect.py) for new words
T = dict(m2["T"])
newT = [w for w in W_all if w not in T]
if newT:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    def target(w):
        fs = {" " + w, w, " " + w.lower(), w.lower(), " " + w.lower().capitalize(), w.lower().capitalize()}
        ids = sorted({t[0] for t in tok(sorted(fs), add_special_tokens=False).input_ids if len(t) == 1})
        u = U[torch.tensor(ids, device=dev)].float()
        return ((u / u.norm(dim=-1, keepdim=True)).mean(0) / g).cpu()
    chk = [w for w in list(m2["T"])[:200]]
    err = max(float((target(w) - m2["T"][w].float()).abs().max()) for w in chk)
    log(f"target recipe check on 200 al2 words: max abs diff {err:.2e}")
    assert err < 1e-3
    for w in newT:
        T[w] = target(w)
TW = torch.stack([T[w] for w in W_all]).to(dev).float()
wi = torch.tensor([widx[w] for w in words_u], device=dev)
rank_t = torch.tensor(rank_u)
rshuf_t = torch.tensor(rshuf_u)
bins_t = torch.where(rank_t > 10, 0, torch.where(rshuf_t <= 10, 1, 2))  # 0 NOT, 1 GIVE, 2 SUB
BIN = {0: "NOT", 1: "GIVE", 2: "SUB"}


def T_(xs):
    return torch.tensor(sorted(set(xs)), dtype=torch.long)


S = {}
S["A2_TR"] = T_(i for i in range(N2) if m2["split"][i] == "train")
S["A2_VAL"] = T_(i for i in range(N2) if m2["split"][i] == "val")
S["T1"] = T_(i for i in range(N2) if m2["split"][i] == "test")
S["T1_nameable"] = T_(i for i in range(N2) if m2["split"][i] == "test" and judged.get(m2["words"][i], {}).get("keep"))
if have3:
    for name, sp in (("A3_TR", "train"), ("A3_VAL", "val"), ("T2", "test"), ("T3", "near")):
        S[name] = T_(u for u, r in zip(u3, rows3) if r["split"] == sp)
    newtr = S["A3_TR"][S["A3_TR"] >= N2]
    S["M_b"] = S["A3_TR"]
    S["M_c"] = torch.cat([S["A2_TR"], newtr])
    S["M_d"] = S["A3_TR"][rank_t[S["A3_TR"]] <= 10]
    S["M_e"] = torch.cat([S["A2_TR"], newtr[rank_t[newtr] <= 10]])
    # al3 view: row -> (type, style, bin) for T2/T3 breakdowns (al2-origin rows appear once per al3 row)
    type_u = {}
    for u, r in zip(u3, rows3):
        type_u[u] = r["type"]
S["M_a"] = S["A2_TR"]
VALSET = {"a": "A2_VAL", "b": "A3_VAL", "c": "A3_VAL", "d": "A3_VAL", "e": "A3_VAL"}
EVSETS = ["A2_VAL", "T1"] + (["A3_VAL", "T2", "T3"] if have3 else [])
EV = torch.unique(torch.cat([S[k] for k in EVSETS]))
pos = torch.full((NU,), -1, dtype=torch.long)
pos[EV] = torch.arange(len(EV))
info = {k: dict(rows=len(v), words=len(set(wi[v.to(dev)].tolist()))) for k, v in S.items()}
log(f"rows al2 {N2} new {NN}; words {len(W_all)}; sets " + json.dumps(info))
Sd = {k: v.to(dev) for k, v in S.items()}


# ---------------- copied from scripts/al2/fit.py ----------------
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


def pack(m):
    mx, W, my = m
    sc = W.abs().max().item()
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.float().cpu())
# -----------------------------------------------------------------


JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}
RK = defaultdict(dict)  # RK[reader][l] = ranks over EV rows (reader: "J++", "M|x", "mix0.3|x")
choice = defaultdict(dict)
maps_save = defaultdict(dict)
EVd = EV.to(dev)
for l in READ:
    t0 = time.time()
    X = torch.load(f"{BIG2}/X_L{l}.pt")
    if have3:
        X = torch.cat([X, torch.load(f"{BIG}/X_L{l}.pt")])
    X = X.to(dev).float()
    xe = X[EVd]
    je = xe @ JPP[l].T
    RK["J++"][l] = ranks_of(je, wi[EVd]).cpu()
    for mname in MAPS:
        z, cfg, m = choose(X, Sd[f"M_{mname}"], Sd[VALSET[mname]])
        choice[mname][l] = dict(cfg=cfg, val_top10=z[0], val_top1=round(z[1], 4))
        me = apply(m, xe)
        RK[f"M|{mname}"][l] = ranks_of(me, wi[EVd]).cpu()
        for lam in LAMS:
            RK[f"mix{lam}|{mname}"][l] = ranks_of(unit(je) + lam * unit(me), wi[EVd]).cpu()
        maps_save[mname][l] = pack(m)
        del m, me
    del X, xe, je
    if dev == "cuda":
        torch.cuda.empty_cache()
    T1p = pos[S["T1"]]
    log(f"L{l} ({time.time() - t0:.0f}s) " + " ".join(f"{k}:{choice[k][l]['cfg']}" for k in MAPS) + " | T1 top10 " +
        " ".join(f"{n} {float((RK[n][l][T1p] <= 10).float().mean()):.4f}" for n in ["J++"] + [f"M|{k}" for k in MAPS]))


def any7(r):
    return torch.stack([RK[r][l] for l in READ]).min(0).values


def sub(r, key_rows):
    return r[pos[key_rows]]


def t10(x):
    return float((x <= 10).float().mean())


# lam and best layer per map and domain
DOM = {"al2": "A2_VAL"} | ({"al3": "A3_VAL"} if have3 else {})
lam_of, best_l = defaultdict(dict), defaultdict(dict)
valtab = defaultdict(dict)
for dom, vs in DOM.items():
    best_l[dom]["J++"] = max(READ, key=lambda l: t10(sub(RK["J++"][l], S[vs])))
    for mname in MAPS:
        vm = {lam: t10(sub(any7(f"mix{lam}|{mname}"), S[vs])) for lam in LAMS}
        lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
        lam_of[dom][mname] = lam
        valtab[dom][mname] = dict(mix_any7_by_lam=vm, M_any7=t10(sub(any7(f"M|{mname}"), S[vs])))
        for rd in (f"M|{mname}", f"mix{lam}|{mname}"):
            best_l[dom][rd] = max(READ, key=lambda l: t10(sub(RK[rd][l], S[vs])))
log(f"lam {dict(lam_of)}; best layers {dict(best_l)}")


def readers(dom):
    out = {"J++": "J++"}
    for mname in MAPS:
        out[f"M_{mname}"] = f"M|{mname}"
        out[f"mix_{mname}"] = f"mix{lam_of[dom][mname]}|{mname}"
    return out


def metrics(rd, rows, dom):
    a7 = sub(any7(rd), rows)
    bl = best_l[dom][rd]
    rb = sub(RK[rd][bl], rows)
    return dict(t10_best=round(t10(rb), 4), t1_best=round(float((rb <= 1).float().mean()), 4), best_layer=bl,
                t10_any7=round(t10(a7), 4), t1_any7=round(float((a7 <= 1).float().mean()), 4), n=len(rows),
                words=len(set(wi[rows.to(dev)].tolist())))


groups = {"T1": ("al2", S["T1"]), "T1_nameable": ("al2", S["T1_nameable"])}
if have3:
    groups["T2"] = ("al3", S["T2"])
    T2l = S["T2"].tolist()
    for t in TYPES:
        groups[f"T2_{t}"] = ("al3", T_(u for u in T2l if type_u[u] == t))
    for b, bn in BIN.items():
        groups[f"T2_{bn}"] = ("al3", S["T2"][bins_t[S["T2"]] == b])
    for k in range(6):
        groups[f"T2_{SNAME[k]}"] = ("al3", T_(u for u in T2l if style_u[u] == k))
    groups["T2_newwords"] = ("al3", T_(u for u in T2l if words_u[u] not in role2))
    groups["T2_al2words"] = ("al3", T_(u for u in T2l if words_u[u] in role2))
    groups["T3"] = ("al3", S["T3"])
res = dict(sets=info, choice={k: {str(l): c for l, c in v.items()} for k, v in choice.items()}, lam=lam_of,
           best_layer={k: {r: l for r, l in v.items()} for k, v in best_l.items()}, val=valtab, tables={})
for gname, (dom, rows) in groups.items():
    if len(rows) == 0:
        continue
    res["tables"][gname] = {nm: metrics(rd, rows, dom) for nm, rd in readers(dom).items()}


# ---------------- paired word bootstrap ----------------
def boot(rdA, rdB, rows, n=2000, seed=0):
    a = (sub(any7(rdA), rows) <= 10).float()
    b = (sub(any7(rdB), rows) <= 10).float()
    w = wi[rows.to(dev)].cpu()
    uw, inv = torch.unique(w, return_inverse=True)
    cnt = torch.zeros(len(uw)).index_add_(0, inv, torch.ones(len(rows)))
    dw = torch.zeros(len(uw)).index_add_(0, inv, b - a)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(uw), (n, len(uw)), generator=gen)
    diffs = dw[idx].sum(1) / cnt[idx].sum(1)
    obs = float((b - a).mean())
    lo, hi = torch.quantile(diffs, torch.tensor([0.025, 0.975])).tolist()
    p = 2 * min(float((diffs <= 0).float().mean()), float((diffs >= 0).float().mean()))
    return dict(diff=round(obs, 4), lo=round(lo, 4), hi=round(hi, 4), p=round(min(1.0, p), 4), words=len(uw),
                n=len(rows))


res["bootstrap"] = {}
for gname in ["T1_nameable", "T1", "T2", "T3"]:
    if gname not in groups:
        continue
    dom, rows = groups[gname]
    rd = readers(dom)
    for B in ("c", "b", "d", "e"):
        if B in MAPS and "a" in MAPS:
            for kind in ("M", "mix"):
                res["bootstrap"][f"{gname} {kind} {B}-a"] = boot(rd[f"{kind}_a"], rd[f"{kind}_{B}"], rows)
for k, v in res["bootstrap"].items():
    log(f"boot {k}: {v}")
for gname, tab in res["tables"].items():
    log(f"{gname}: " + " | ".join(f"{nm} {m['t10_best']:.3f}/{m['t1_best']:.3f}/{m['t10_any7']:.3f}" for nm, m in tab.items()))
tag = args.tag
os.makedirs(RES, exist_ok=True)
json.dump(res, open(f"{RES}/fit_results{tag}.json", "w"), indent=1, default=str)
torch.save(dict(RK=dict(RK), EV=EV, layers=READ, sets=S), f"{BIG}/fit_ranks{tag}.pt")
if READ == LAYERS:
    for mname in MAPS:
        torch.save(maps_save[mname], f"{BIG}/M_{mname}{tag}.pt")
log(f"done {time.time() - t00:.0f}s")
print("FIT_DONE")
