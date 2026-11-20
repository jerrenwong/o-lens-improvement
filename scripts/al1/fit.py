"""al1 step 3: learned concept->word maps (M1-M4), chosen on 200 validation words, tested on 200 unseen test words.
M maps are LEARNED (ridge), not Jacobians. Outputs live in the final basis (t/g); ranking = (v*g) @ W_U^T with the
benchmark's punctuation filter (tokens with no letter/digit excluded). Hit = rank of best surface form <= k
(surface form = any vocab token whose stripped lower-case string equals the word, lower-cased).
Writes {BIG}/fit_results.json, {BIG}/M_best.pt, {BIG}/M_shuf.pt (e2 map format) and {BIG}/test_ranks.pt."""
import json, os, random, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al1_common import R, BIG, log
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
MN = "Qwen/Qwen3.5-9B"
GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
RANKS = [64, 256, 1024]
READ = [4, 8, 12, 16, 20, 24, 28]
KS = [1, 2, 4, 8]
SIZES = [250, 500, 1000, 2000]
t00 = time.time()

meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
LAYERS = meta["layers"]
words_s, split = meta["words"], meta["split"]
role = meta["role"]
g = meta["g"].to(dev)
U = torch.load(f"{BIG}/U.pt").to(dev)  # bf16 [V, d]
V, d = U.shape
strs = meta["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
norm_s = [s.strip().lower() for s in strs]
by_norm = defaultdict(list)
for i, s in enumerate(norm_s):
    by_norm[s].append(i)
W_all = sorted(role)  # all words
widx = {w: i for i, w in enumerate(W_all)}
forms = [by_norm[w.lower()] for w in W_all]
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
log(f"vocab {V}; excluded (no alnum) {int(excl.sum())}; surface forms per word max {mf}")
TW = torch.stack([meta["T"][w] for w in W_all]).to(dev).float()  # word targets, final basis
wi = torch.tensor([widx[w] for w in words_s], device=dev)
sp = {r: torch.tensor([s == r for s in split], device=dev).nonzero()[:, 0] for r in ("train", "val", "test")}
wid = wi
tr_words = sorted({widx[w] for w, r in role.items() if r == "train"})
log("sentences " + " ".join(f"{k} {len(v)}" for k, v in sp.items()) + f"; train words {len(tr_words)}")


def ranks_of(vecs, wid, bs=512):
    """rank (1 = top) of the word's best surface form under (v*g) @ U^T with the punctuation filter."""
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
    return dict(top10=round(float((r <= 10).float().mean()), 4), top1=round(float((r <= 1).float().mean()), 4))


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class Ridge:
    """ridge with bias: rows X -> Y, closed form via eigh; lambda = c * mean eigenvalue."""

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


def class_stats(X, idx, wid_tr):
    """class means over the given sentence rows (per word), within-class covariance."""
    ws = wid[idx]
    uw, inv = torch.unique(ws, return_inverse=True)
    Mn = torch.zeros(len(uw), d, device=dev).index_add_(0, inv, X[idx])
    cnt = torch.zeros(len(uw), device=dev).index_add_(0, inv, torch.ones(len(idx), device=dev))
    Mn /= cnt[:, None]
    Rw = (X[idx] - Mn[inv]).double()
    Sw = Rw.T @ Rw / len(idx)
    return uw, Mn, Sw


def whitener(Sw, f):
    s, Q = torch.linalg.eigh(Sw)
    s = s.clamp_min(0)
    eps = f * s.mean()
    return (Q @ torch.diag((s + eps) ** -0.5) @ Q.T).float()


def fit_maps(X, tr_idx, Tfn, cfg_only=None):
    """yield (name, cfg, (mx, W, my)) for every map config. Tfn(word ids) -> targets."""
    want = lambda n: cfg_only is None or n in cfg_only  # noqa: E731
    if want("M1"):
        r = Ridge(X[tr_idx]); r.setY(Tfn(wid[tr_idx]))
        for c in GRID:
            yield ("M1", dict(c=c), (r.mx, r.W(c), r.my))
        del r
    if not (want("M2") or want("M3") or want("M4")):
        return
    uw, Mn, Sw = class_stats(X, tr_idx, None)
    Tm = Tfn(uw)
    if want("M2"):
        r = Ridge(Mn); r.setY(Tm)
        for c in GRID:
            yield ("M2", dict(c=c), (r.mx, r.W(c), r.my))
        del r
    if want("M3") or want("M4"):
        for f in EPSF:
            P = whitener(Sw, f)
            r = Ridge(Mn @ P); r.setY(Tm)
            for c in GRID:
                Wc = P @ r.W(c)
                if want("M3"):
                    yield ("M3", dict(f=f, c=c), (Mn.mean(0), Wc, r.my))
                if want("M4"):
                    Us, Ss, Vh = torch.linalg.svd(Wc, full_matrices=False)
                    for k in RANKS:
                        yield ("M4", dict(f=f, c=c, r=k), (Mn.mean(0), (Us[:, :k] * Ss[:k]) @ Vh[:k], r.my))
            del r, P


def apply(m, x):
    mx, W, my = m
    return (x - mx) @ W + my


JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=MN).jacobians_L_dict_FN.items()}
JL = {int(l): v.to(dev).float() for l, v in
      load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=MN).jacobians_L_dict_FN.items()}


def emap(path):
    m = torch.load(path, map_location="cpu")
    return {int(l): (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items()}


EA, GS = emap(f"{R}/runs/e1/E_9b.pt"), emap(f"{R}/runs/g1/G_strict.pt")
log(f"lens layers: J++ {len(JPP)} J {len(JL)} E {sorted(EA)} G {sorted(GS)}")

rng = random.Random(0)
te_idx = sp["test"]
# k-sentence inputs: per test word, disjoint groups of k sentences
grp = {}
te_by = defaultdict(list)
for i in te_idx.tolist():
    te_by[int(wi[i])].append(i)
for k in KS:
    gl, gw = [], []
    for w, ii in sorted(te_by.items()):
        ii = list(ii); random.Random(k * 1000 + w).shuffle(ii)
        for j in range(len(ii) // k):
            gl.append(ii[j * k:(j + 1) * k]); gw.append(w)
    grp[k] = (gl, torch.tensor(gw, device=dev))
log("k-groups: " + " ".join(f"k{k}:{len(grp[k][0])}" for k in KS))

res = dict(layers={}, choice={}, kmean={}, counts=dict(sent={k: len(v) for k, v in sp.items()}))
test_ranks = {}  # (method, layer) -> ranks on test sentences
val_ranks = {}
kranks = {}  # (k, method, layer) -> ranks on k-groups
best_cfg = {}
saved_maps = {}
for l in LAYERS:
    t0 = time.time()
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    tr = sp["train"]
    Tfn = lambda ids: TW[ids]  # noqa: E731
    xv, yv = X[sp["val"]], wi[sp["val"]]
    sc, top = defaultdict(list), {}
    for name, cfg, m in fit_maps(X, tr, Tfn):
        r = ranks_of(apply(m, xv), yv)
        z = ((r <= 10).float().mean().item(), (r <= 1).float().mean().item(), cfg)
        sc[name].append(z)
        if name not in top or (round(z[0], 6), z[1]) > (round(top[name][0], 6), top[name][1]):
            top[name] = (z[0], z[1], cfg, m, r)
        del m
    xt, yt = X[te_idx], wi[te_idx]
    outs_t = {}
    chosen = {}
    for name in ("M1", "M2", "M3", "M4"):
        b = top[name]
        chosen[name] = dict(cfg=b[2], val_top10=round(b[0], 4), val_top1=round(b[1], 4),
                            grid=[(z[2], round(z[0], 4)) for z in sc[name]])
        best_cfg[(name, l)] = b[2]
        val_ranks[(name, l)] = b[4]
        outs_t[name] = lambda x, m=b[3]: apply(m, x)  # noqa: E731
        if l in READ:
            saved_maps[(name, l)] = tuple(t.cpu() for t in b[3])
    del sc, top
    outs_t["J++"] = lambda x: x @ JPP[l].T  # noqa: E731
    outs_t["J"] = lambda x: x @ JL[l].T  # noqa: E731
    outs_t["logit"] = lambda x: x  # noqa: E731
    if l in EA:
        outs_t["E"] = lambda x: apply(EA[l], x)  # noqa: E731
    if l in GS:
        outs_t["G_g1"] = lambda x: apply(GS[l], x)  # noqa: E731
    val_ranks[("J++", l)] = ranks_of(xv @ JPP[l].T, yv)
    row = {}
    for name, fn in outs_t.items():
        r = ranks_of(fn(xt), yt)
        test_ranks[(name, l)] = r.cpu()
        row[name] = acc(r)
        for k in KS:
            gl, gw = grp[k]
            xm = torch.stack([X[ii].mean(0) for ii in gl])
            kranks[(k, name, l)] = ranks_of(fn(xm), gw).cpu()
    res["layers"][l] = row
    res["choice"][l] = chosen
    log(f"L{l:2d} ({time.time() - t0:.0f}s) choices " +
        " ".join(f"{n}:{chosen[n]['cfg']} val{chosen[n]['val_top10']}" for n in chosen) +
        " | test top10 " + " ".join(f"{n} {row[n]['top10']:.3f}" for n in row))
    # store per-layer callable for mixes later: keep maps for M at readout layers only (memory)
    del X
    torch.cuda.empty_cache()


def anylayer(store, name, key=None):
    rs = [store[(name, l)] if key is None else store[(key, name, l)] for l in READ if
          ((name, l) if key is None else (key, name, l)) in store]
    return torch.stack(rs).min(0).values if rs else None


# best M on validation any-layer
vbest = {n: float((anylayer(val_ranks, n) <= 10).float().mean()) for n in ("M1", "M2", "M3", "M4")}
BEST = max(vbest, key=vbest.get)
res["val_anylayer_top10"] = vbest
res["val_anylayer_top10"]["J++"] = float((anylayer(val_ranks, "J++") <= 10).float().mean())
res["best_M"] = BEST
blay = max(READ, key=lambda l: float((val_ranks[(BEST, l)] <= 10).float().mean()))
jlay = max(READ, key=lambda l: float((val_ranks[("J++", l)] <= 10).float().mean()))
res["best_layer"] = dict(M=blay, Jpp=jlay)
log(f"best M on val (any layer top-10): {vbest} -> {BEST}; best layer {blay}; J++ best val layer {jlay}")

# mixes unit(J++ h) + lam unit(M h): lam chosen on val any-layer
mix_rank_v, mix_rank_t, mix_k = {}, {}, {}
for l in READ:
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    m = tuple(t.to(dev) for t in saved_maps[(BEST, l)])
    for lam in (0.3, 1.0):
        f = lambda x: unit(x @ JPP[l].T) + lam * unit(apply(m, x))  # noqa: E731
        mix_rank_v[(f"mix{lam}", l)] = ranks_of(f(X[sp["val"]]), wi[sp["val"]])
        mix_rank_t[(f"mix{lam}", l)] = ranks_of(f(X[te_idx]), wi[te_idx]).cpu()
        for k in KS:
            gl, gw = grp[k]
            xm = torch.stack([X[ii].mean(0) for ii in gl])
            mix_k[(k, f"mix{lam}", l)] = ranks_of(f(xm), gw).cpu()
    del X
vm = {lam: float((anylayer(mix_rank_v, f"mix{lam}") <= 10).float().mean()) for lam in (0.3, 1.0)}
LAM = max(vm, key=lambda z: (vm[z], -z))
res["mix_val"] = vm; res["mix_lam"] = LAM
for l in READ:
    test_ranks[("mix", l)] = mix_rank_t[(f"mix{LAM}", l)]
    res["layers"][l]["mix"] = acc(test_ranks[("mix", l)])
    for k in KS:
        kranks[(k, "mix", l)] = mix_k[(k, f"mix{LAM}", l)]
log(f"mix lam val {vm} -> {LAM}")

NAMES = ["M1", "M2", "M3", "M4", "J++", "J", "logit", "G_g1", "E", "mix"]
res["any_layer"] = {n: acc(anylayer(test_ranks, n)) for n in NAMES}
res["best_single_layer_test"] = {n: max(((l, acc(test_ranks[(n, l)])) for l in READ if (n, l) in test_ranks),
                                        key=lambda z: z[1]["top10"]) for n in NAMES}
for k in KS:
    res["kmean"][k] = {n: dict(any=acc(anylayer(kranks, n, k)),
                               best=max(((l, acc(kranks[(k, n, l)])) for l in READ if (k, n, l) in kranks),
                                        key=lambda z: z[1]["top10"]),
                               n=len(grp[k][0])) for n in NAMES}
log("any-layer test: " + " ".join(f"{n} {res['any_layer'][n]}" for n in NAMES))

# complementarity (any layer, top-10) J++ vs best M
a = anylayer(test_ranks, "J++") <= 10
b = anylayer(test_ranks, BEST) <= 10
res["complement"] = dict(both=int((a & b).sum()), jpp_only=int((a & ~b).sum()), M_only=int((~a & b).sum()),
                         neither=int((~a & ~b).sum()), n=len(a), union=round(float((a | b).float().mean()), 4))
a1 = test_ranks[("J++", jlay)] <= 10
b1 = test_ranks[(BEST, blay)] <= 10
res["complement_bestlayer"] = dict(both=int((a1 & b1).sum()), jpp_only=int((a1 & ~b1).sum()),
                                   M_only=int((~a1 & b1).sum()), neither=int((~a1 & ~b1).sum()))
# per word: share of words where either hits in >= 1 sentence
log(f"complement {res['complement']} best-layer {res['complement_bestlayer']}")

# examples
ex_ids = random.Random(5).sample(te_idx.tolist(), 20)
ex = []
Xb = torch.load(f"{BIG}/X_L{blay}.pt")[ex_ids].to(dev).float()
Xj = torch.load(f"{BIG}/X_L{jlay}.pt")[ex_ids].to(dev).float()
tm = topk_tokens(apply(tuple(t.to(dev) for t in saved_maps[(BEST, blay)]), Xb))
tj = topk_tokens(Xj @ JPP[jlay].T)
pos = {i: j for j, i in enumerate(te_idx.tolist())}
for j, i in enumerate(ex_ids):
    ex.append(dict(word=words_s[i], sent=meta["sents"][i],
                   M=[strs[t].strip() for t in tm[j].tolist()], M_rank=int(test_ranks[(BEST, blay)][pos[i]]),
                   J=[strs[t].strip() for t in tj[j].tolist()], J_rank=int(test_ranks[("J++", jlay)][pos[i]])))
res["examples"] = ex

# learning curve for BEST (configs f, r fixed as chosen per layer; c re-chosen on val per size)
tw = list(tr_words); random.Random(0).shuffle(tw)
lc = {}
lc_ranks = defaultdict(dict)
for l in READ:
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    cfg = best_cfg[(BEST, l)]
    for n in SIZES:
        keep = torch.tensor(sorted(tw[:n]), device=dev)
        tr = sp["train"][torch.isin(wi[sp["train"]], keep)]
        e0, r0 = EPSF, RANKS
        EPSF = [cfg["f"]] if "f" in cfg else EPSF
        RANKS = [cfg["r"]] if "r" in cfg else RANKS
        bestv, bm = -1, None
        for name, cf, m in fit_maps(X, tr, lambda ids: TW[ids], cfg_only={BEST}):
            if name != BEST:
                continue
            v = float((ranks_of(apply(m, X[sp["val"]]), wi[sp["val"]]) <= 10).float().mean())
            if v > bestv:
                bestv, bm = v, m
        EPSF, RANKS = e0, r0
        lc_ranks[n][l] = ranks_of(apply(bm, X[te_idx]), wi[te_idx]).cpu()
        del bm
    del X
    torch.cuda.empty_cache()
for n in SIZES:
    lc[n] = dict(any=acc(torch.stack([lc_ranks[n][l] for l in READ]).min(0).values),
                 at_best_layer=acc(lc_ranks[n][blay]))
res["learning_curve"] = lc
log(f"learning curve {lc}")

# shuffled-label control map (word targets permuted across training words), same per-layer config
perm_words = list(tr_words)
pw = perm_words[:]; random.Random(1).shuffle(pw)
PM = torch.arange(len(W_all), device=dev)
PM[torch.tensor(perm_words, device=dev)] = torch.tensor(pw, device=dev)
shuf = {}
for l in READ:
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    cfg = best_cfg[(BEST, l)]
    e0, r0 = EPSF, RANKS
    EPSF = [cfg["f"]] if "f" in cfg else EPSF
    RANKS = [cfg["r"]] if "r" in cfg else RANKS
    m = [m for name, cf, m in fit_maps(X, sp["train"], lambda ids: TW[PM[ids]], cfg_only={BEST})
         if name == BEST and cf == cfg][0]
    EPSF, RANKS = e0, r0
    shuf[l] = m
    test_ranks[("shuf", l)] = ranks_of(apply(m, X[te_idx]), wi[te_idx]).cpu()
    del X
res["any_layer"]["M_shuf"] = acc(anylayer(test_ranks, "shuf"))
log(f"shuffled-label map any-layer test {res['any_layer']['M_shuf']}")


def pack(m):
    mx, W, my = m
    sc = W.abs().max().item()
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.float().cpu())


torch.save({l: pack(saved_maps[(BEST, l)]) for l in READ}, f"{BIG}/M_best.pt")
torch.save({l: pack(shuf[l]) for l in READ}, f"{BIG}/M_shuf.pt")
torch.save(dict(test=test_ranks, k=kranks, lc=dict(lc_ranks)), f"{BIG}/test_ranks.pt")
res["best_cfg"] = {f"{n}|{l}": c for (n, l), c in best_cfg.items()}
json.dump(res, open(f"{BIG}/fit_results.json", "w"), indent=1, default=str)
log(f"done {time.time() - t00:.0f}s")
