"""sub2 step 2a: refit al2's M3 map (code copied from scripts/al2/fit.py: class_stats, whitener, Ridge, fit_M3, choose,
ranks_of, (eps, c) grid) on subsets of the TRAIN-word sentences defined by step-1 bins, and store val/test ranks.
Bins (from .hf27/sub2/al2_ranks.pt): NOT rank > 10; GIVE rank <= 10 & rank_shuf <= 10; SUB rank <= 10 & rank_shuf > 10.
Subsets: ALL, SUB, INF (=SUB+GIVE), GIVE, NOT, RSUB{0,1,2} (per word as many random train sentences as it has SUB),
RINF{0,1,2} (same, matched to INF). (eps, c) per layer and subset chosen on ALL validation sentences (all styles),
exactly as M_all. Readers: M alone, released J++ alone, unit(J++ h) + lam unit(M h) for lam in {0.1, 0.3, 1.0}.
Writes OUT/fit_ranks.pt (VR/TR rank tensors), OUT/fit_choice.json, OUT/M_<name>.pt (e2 map format) for SAVE.
Usage: fit2.py [--layers 4,8,...] [--dev cuda|cpu] [--fake] [--tiny K] [--out DIR]"""
import argparse, json, os, random, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
_argv, sys.argv = sys.argv, sys.argv[:1]  # al2_common imports g1_gen, which parses sys.argv
from al2_common import R, BIG, BIG1, log, free_mb  # noqa: E402
sys.argv = _argv
from workspace_lens.utils import load_lens_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="4,8,12,16,20,24,28")
ap.add_argument("--dev", default="cuda")
ap.add_argument("--fake", action="store_true", help="random bins (logic test only)")
ap.add_argument("--tiny", type=int, default=0, help="keep only K train words and 300 val/test rows (logic test)")
ap.add_argument("--out", default=f"{R}/.hf27/sub2")
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = args.dev
if dev == "cpu":
    torch.set_num_threads(64)
OUT = args.out
os.makedirs(OUT, exist_ok=True)
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [int(x) for x in args.layers.split(",")]
LAMS = [0.1, 0.3, 1.0]
SAVE = ["ALL", "SUB", "INF", "RSUB0", "RINF0"]
if dev == "cuda":
    while free_mb() < 36000:
        log(f"waiting for GPU: free {free_mb()} MiB < 36000")
        time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(0.42)
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style, role = meta["words"], meta["split"], meta["style"], meta["role"]
g = meta["g"].to(dev)
U = torch.load(f"{BIG1}/U.pt").to(dev)
if dev == "cpu":
    U = U.float()
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
mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
TW = torch.stack([meta["T"][w] for w in W_all]).to(dev).float()
wi = torch.tensor([widx[w] for w in words_s], device=dev)
sty = torch.tensor(style, device=dev)
spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("train", "val", "test")}
N = len(words_s)

if args.fake:
    bins = torch.randint(0, 3, (N,), generator=torch.Generator().manual_seed(0))
else:
    L = torch.load(f"{OUT}/al2_ranks.pt")
    assert bool(L["labelled"].all())
    rk, rs = L["rank"], L["rank_shuf"]
    bins = torch.where(rk > 10, 0, torch.where(rs <= 10, 1, 2))  # 0 NOT, 1 GIVE, 2 SUB
bins = bins.to(dev)
TRN = spl["train"].nonzero()[:, 0]
VAL = spl["val"].nonzero()[:, 0]
TE = spl["test"].nonzero()[:, 0]
if args.tiny:
    keepw = torch.tensor(sorted(set(wi[TRN].tolist()))[: args.tiny], device=dev)
    TRN = TRN[torch.isin(wi[TRN], keepw)]
    VAL, TE = VAL[:300], TE[:300]
log(f"sentences {N}; train {len(TRN)} val {len(VAL)} test {len(TE)}; words {len(W_all)}; vocab {V}; dev {dev}")


def matched(ref_mask, seed):
    """per train word: as many random train sentences (any bin) as it has in ref_mask; words with 0 dropped."""
    rng = random.Random(seed)
    byw = defaultdict(list)
    for i, w in zip(TRN.tolist(), wi[TRN].tolist()):
        byw[w].append(i)
    cnt = defaultdict(int)
    for w in wi[TRN[ref_mask[TRN]]].tolist():
        cnt[w] += 1
    out = []
    for w in sorted(cnt):
        out += rng.sample(byw[w], cnt[w])
    return torch.tensor(sorted(out), device=dev)


SUBM = bins == 2
INFM = bins >= 1
SUBSETS = {"ALL": TRN, "SUB": TRN[SUBM[TRN]], "INF": TRN[INFM[TRN]], "GIVE": TRN[(bins == 1)[TRN]],
           "NOT": TRN[(bins == 0)[TRN]]}
for s in range(3):
    SUBSETS[f"RSUB{s}"] = matched(SUBM, 100 + s)
    SUBSETS[f"RINF{s}"] = matched(INFM, 200 + s)
info = {k: dict(n_sent=len(v), n_words=len(set(wi[v].tolist()))) for k, v in SUBSETS.items()}
log("subsets " + json.dumps(info))
for s in range(3):  # matched-control sanity: same words, same per-word counts
    for a, b_ in ((f"RSUB{s}", "SUB"), (f"RINF{s}", "INF")):
        assert torch.equal(torch.sort(wi[SUBSETS[a]]).values, torch.sort(wi[SUBSETS[b_]]).values), a


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
VR = defaultdict(dict)
TR = defaultdict(dict)
choice = defaultdict(dict)
maps_save = defaultdict(dict)
for l in READ:
    t0 = time.time()
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    jv, jt = X[VAL] @ JPP[l].T, X[TE] @ JPP[l].T
    VR["J++"][l] = ranks_of(jv, wi[VAL]).cpu()
    TR["J++"][l] = ranks_of(jt, wi[TE]).cpu()
    for name, tr in SUBSETS.items():
        z, cfg, m = choose(X, tr, VAL)
        choice[name][l] = dict(cfg=cfg, val_top10=z[0])
        mv, mt = apply(m, X[VAL]), apply(m, X[TE])
        VR[f"M|{name}"][l] = ranks_of(mv, wi[VAL]).cpu()
        TR[f"M|{name}"][l] = ranks_of(mt, wi[TE]).cpu()
        for lam in LAMS:
            VR[f"mix{lam}|{name}"][l] = ranks_of(unit(jv) + lam * unit(mv), wi[VAL]).cpu()
            TR[f"mix{lam}|{name}"][l] = ranks_of(unit(jt) + lam * unit(mt), wi[TE]).cpu()
        if name in SAVE:
            maps_save[name][l] = pack(m)
        del m, mv, mt
    log(f"L{l} ({time.time() - t0:.0f}s) test top10 " +
        " ".join(f"{n} {float((TR[n][l] <= 10).float().mean()):.3f}" for n in ["J++"] + [f"M|{k}" for k in SUBSETS]) +
        " | cfg " + " ".join(f"{k}:{choice[k][l]['cfg']['f']},{choice[k][l]['cfg']['c']}" for k in SUBSETS))
    del X, jv, jt
    if dev == "cuda":
        torch.cuda.empty_cache()

torch.save(dict(VR=dict(VR), TR=dict(TR), val_rows=VAL.cpu(), te_rows=TE.cpu(), bins=bins.cpu(), layers=READ,
                subsets={k: v.cpu() for k, v in SUBSETS.items()}), f"{OUT}/fit_ranks.pt")
json.dump(dict(choice={k: {str(l): c for l, c in v.items()} for k, v in choice.items()}, subsets=info, layers=READ),
          open(f"{OUT}/fit_choice.json", "w"), indent=1)
if READ == [4, 8, 12, 16, 20, 24, 28]:
    for name in SAVE:
        torch.save(maps_save[name], f"{OUT}/M_{name}.pt")
log(f"done {time.time() - t00:.0f}s")
print("FIT2_DONE")
