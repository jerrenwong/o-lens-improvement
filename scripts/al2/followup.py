"""al2 follow-up: fairer J++ baselines on al2's unseen-word test sentences (no benchmark data).
J versions: (1) released J++, (2) refit experts + benchmark-fit weights, (3) same experts + weights fitted on 4,000
generated sentences (w1). M_all loaded from .hf27/al2/M_all.pt; M_noS3 refitted with its chosen config (eps .1, c 1).
Mix unit(J h) + lam unit(M h), lam in {0.1, 0.3, 1.0} chosen per J version on validation words (any-of-7 top-10;
for M_noS3 on validation non-S3 sentences). Reported overall, by style, and on test words not among W-train words."""
import json, os, sys
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al2_common import R, BIG, BIG1, log, wait_gpu
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
wait_gpu(30000)
READ = [4, 8, 12, 16, 20, 24, 28]
LAMS = [0.1, 0.3, 1.0]
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
words_s, split, style, role = meta["words"], meta["split"], meta["style"], meta["role"]
g = meta["g"].to(dev)
U = torch.load(f"{BIG1}/U.pt").to(dev)
strs = torch.load(f"{BIG1}/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
by_norm = defaultdict(list)
for i, s in enumerate(strs):
    by_norm[s.strip().lower()].append(i)
W_all = sorted(role); widx = {w: i for i, w in enumerate(W_all)}
forms = [by_norm[w.lower()] for w in W_all]; mf = max(len(f) for f in forms)
FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
TW = torch.stack([meta["T"][w] for w in W_all]).to(dev).float()
wi = torch.tensor([widx[w] for w in words_s], device=dev)
sty = torch.tensor(style, device=dev)
spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("train", "val", "test")}
def rows(r, ss): return (spl[r] & torch.isin(sty, torch.tensor(sorted(ss), device=dev))).nonzero()[:, 0]
TE, VA, VA_n3 = rows("test", range(5)), rows("val", range(5)), rows("val", {0, 1, 3, 4})
TR_n3 = rows("train", {0, 1, 3, 4})
te_sty = sty[TE].cpu()
# W-train words (w1's generated-sentence fit set)
G = json.load(open(f"{R}/data/evals_gen/lens-eval-genword.json"))["items"]
wtr = set(json.load(open(f"{R}/data/evals_gen/splits.json"))["W-train"])
wtrain_words = {c.lower() for it in G if it["name"] in wtr for c in it["intermediates"]}
te_words = [words_s[i] for i in TE.tolist()]
fresh = torch.tensor([w.lower() not in wtrain_words for w in te_words])
n_fresh_words = len({w for w, f in zip(te_words, fresh.tolist()) if f})
log(f"test sentences {len(TE)}; test words {len(set(te_words))}; not in W-train: {n_fresh_words} words, {int(fresh.sum())} sentences")


def ranks_of(vecs, wid, bs=512):
    out = []
    for i in range(0, vecs.shape[0], bs):
        lg = ((vecs[i:i + bs].float() * g).to(U.dtype) @ U.T).float().masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid[i:i + bs]]).max(1).values
        out.append((lg > best[:, None]).sum(1) + 1)
    return torch.cat(out).cpu()


def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def class_stats(X, idx):
    uw, inv = torch.unique(wi[idx], return_inverse=True)
    Mn = torch.zeros(len(uw), X.shape[1], device=dev).index_add_(0, inv, X[idx])
    cnt = torch.zeros(len(uw), device=dev).index_add_(0, inv, torch.ones(len(idx), device=dev))
    Mn /= cnt[:, None]
    Sw = torch.zeros(X.shape[1], X.shape[1], device=dev, dtype=torch.float64)
    for i in range(0, len(idx), 16384):
        Rw = X[idx[i:i + 16384]] - Mn[inv[i:i + 16384]]
        Sw += (Rw.T @ Rw).double()
    return uw, Mn, Sw / len(idx)


def fit_noS3(X, f=0.1, c=1.0):
    uw, Mn, Sw = class_stats(X, TR_n3)
    s, Q = torch.linalg.eigh(Sw); s = s.clamp_min(0)
    P = (Q @ torch.diag((s + f * s.mean()) ** -0.5) @ Q.T).float()
    Z = Mn @ P; mz = Z.mean(0); Zc = (Z - mz).double()
    e, Qz = torch.linalg.eigh(Zc.T @ Zc); e = e.clamp_min(0)
    Y = TW[uw]; my = Y.mean(0)
    W = (Qz @ ((Qz.T @ (Zc.T @ (Y - my).double())) / (e + c * e.mean())[:, None])).float()
    return Mn.mean(0), P @ W, my


def lensd(p):
    return {int(l): v for l, v in load_lens_file(f"{R}/{p}", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}


JP = {"J1_released": "artifacts/baselines/jpp9b_prefit.pt", "J2_benchfit": "runs/r2_9b/A/A_full/jpp_A_full.pt",
      "J3_genfit": "runs/w1/jpp_Afull_Wtrain.pt"}
J = {k: lensd(p) for k, p in JP.items()}
for k in J:
    assert all(l in J[k] for l in READ), (k, sorted(J[k]))
MA = torch.load(f"{BIG}/M_all.pt", map_location="cpu")
VR, TR = defaultdict(dict), defaultdict(dict)
for l in READ:
    X = torch.load(f"{BIG}/X_L{l}.pt").to(dev).float()
    v = MA[l]; m = (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
    mn3 = fit_noS3(X)
    M = {"M_all": m, "M_noS3": mn3}
    out = {}
    for mk, (mx, W, my) in M.items():
        out[mk] = {r: (X[ix] - mx) @ W + my for r, ix in (("va", VA), ("va3", VA_n3), ("te", TE))}
        TR[mk][l] = ranks_of(out[mk]["te"], wi[TE])
    for jk, Jd in J.items():
        Jl = Jd[l].to(dev).float()
        jv = {r: X[ix] @ Jl.T for r, ix in (("va", VA), ("va3", VA_n3), ("te", TE))}
        TR[jk][l] = ranks_of(jv["te"], wi[TE])
        for lam in LAMS:
            VR[f"{jk}+M_all@{lam}"][l] = ranks_of(unit(jv["va"]) + lam * unit(out["M_all"]["va"]), wi[VA])
            TR[f"{jk}+M_all@{lam}"][l] = ranks_of(unit(jv["te"]) + lam * unit(out["M_all"]["te"]), wi[TE])
            VR[f"{jk}+M_noS3@{lam}"][l] = ranks_of(unit(jv["va3"]) + lam * unit(out["M_noS3"]["va3"]), wi[VA_n3])
            TR[f"{jk}+M_noS3@{lam}"][l] = ranks_of(unit(jv["te"]) + lam * unit(out["M_noS3"]["te"]), wi[TE])
        del Jl
    log(f"L{l} " + " ".join(f"{k} {float((TR[k][l] <= 10).float().mean()):.3f}" for k in list(J) + ["M_all", "M_noS3"]))
    del X
    torch.cuda.empty_cache()


def any7(d): return torch.stack([d[l] for l in READ]).min(0).values


def a(r): return [round(float((r <= 10).float().mean()), 4), round(float((r <= 1).float().mean()), 4), int(len(r))]


res = dict(n_test_sent=len(TE), n_fresh_words=n_fresh_words, n_fresh_sent=int(fresh.sum()), lam={}, val={})
for jk in J:
    for mk in ("M_all", "M_noS3"):
        vm = {lam: float((any7(VR[f"{jk}+{mk}@{lam}"]) <= 10).float().mean()) for lam in LAMS}
        lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
        res["lam"][f"{jk}+{mk}"] = lam; res["val"][f"{jk}+{mk}"] = vm
        TR[f"{jk}+{mk}"] = TR[f"{jk}+{mk}@{lam}"]
NAMES = list(J) + ["M_all"] + [f"{jk}+M_all" for jk in J]
res["T_all"] = {}
for n in NAMES:
    r = any7(TR[n])
    res["T_all"][n] = dict(all=a(r), fresh=a(r[fresh]), **{f"S{k + 1}": a(r[te_sty == k]) for k in range(5)},
                           **{f"fresh_S{k + 1}": a(r[fresh & (te_sty == k)]) for k in range(5)})
s3 = te_sty == 2
res["T_S3"] = {n: dict(all=a(any7(TR[n])[s3]), fresh=a(any7(TR[n])[s3 & fresh]))
               for n in ["J1_released", "J3_genfit", "M_noS3", "M_all", "J1_released+M_noS3", "J3_genfit+M_noS3",
                         "J1_released+M_all", "J3_genfit+M_all"]}
json.dump(res, open(f"{BIG}/followup.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("FOLLOWUP_DONE")
