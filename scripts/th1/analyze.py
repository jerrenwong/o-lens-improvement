"""th1 step 3 (no model): per layer 0..30 decoding among the 200 words (THINK probe, THINK class-mean, SAY via J++,
SAYJ via J-lens, logit lens), geometry of THINK vs SAY vs RAW directions, overlap of hits; save causal directions."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from th1_common import R, BIG, log
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
M = "Qwen/Qwen3.5-9B"
D = torch.load(f"{BIG}/acts.pt", weights_only=False)
X, wi = D["X"], D["wi"].to(dev)
NW = len(D["words"])
tr = torch.tensor([s == "train" for s in D["split"]], device=dev); te = ~tr
g = D["g"].to(dev); Uf = D["Uf"].to(dev); fw = torch.tensor(D["form_word"], device=dev)
uX = D["uX"].to(dev); eX = D["eX"].to(dev)
JPP = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
JL = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
log(f"acts {tuple(X.shape)}; J++ layers {sorted(JPP)[:3]}..{sorted(JPP)[-1]} ({len(JPP)}); J layers {len(JL)}")
kinds = D["kinds"]; kind_of = torch.tensor([sorted(set(kinds)).index(k) for k in kinds], device=dev)
KN = sorted(set(kinds))


def word_scores(h, S):  # h [n,d], S [nf,d] per-form readout rows -> best form per word [n,NW]
    lg = h @ S.T
    return torch.full((h.shape[0], NW), -1e30, device=dev).scatter_reduce(1, fw[None].expand(h.shape[0], -1), lg, "amax")


def ranks(sc, y):
    return (sc > sc.gather(1, y[:, None])).sum(1) + 1


def acc(r):
    return dict(top1=round(float((r <= 1).float().mean()), 4), top5=round(float((r <= 5).float().mean()), 4))


def fit_probe(Z, y, lam, iters=150):
    Wt = torch.zeros(NW, Z.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(NW, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([Wt, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe", history_size=20)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ Wt.T + b, y) + lam * (Wt ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return Wt.detach(), b.detach()


def geom(T, S):
    out = {}
    for cen in (False, True):
        a = T - T.mean(0) if cen else T
        c = S - S.mean(0) if cen else S
        C = torch.nn.functional.normalize(a, dim=1) @ torch.nn.functional.normalize(c, dim=1).T
        eye = torch.eye(NW, device=dev, dtype=torch.bool)
        out["centred" if cen else "raw"] = dict(
            matched=round(float(C[eye].mean()), 4), mismatched=round(float(C[~eye].mean()), 4),
            ident=round(float((C.argmax(1) == torch.arange(NW, device=dev)).float().mean()), 4))
    return out


# held-out fifth of train for the probe strength: 7 of each word's 35 (first 7 in order)
tr_idx = tr.nonzero()[:, 0]
cnt = torch.zeros(NW, dtype=torch.long)
val_mask = torch.zeros_like(tr)
for i in tr_idx.tolist():
    w = int(wi[i])
    if cnt[w] < 7:
        val_mask[i] = True
    cnt[w] += 1
fit_mask = tr & ~val_mask
LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
res, causal = {"layers": {}}, {}
y_te = wi[te]
asked_r = ranks(D["ask_lg"].to(dev)[te], y_te)
res["asked_test"] = acc(asked_r)
log("asked (test):", res["asked_test"])
for L in D["layers"]:
    t0 = time.time()
    h = X[:, L].to(dev).float()
    mu = h[tr].mean(0)
    hc = h - mu
    scale = hc[tr].norm(dim=1).mean()
    A = torch.zeros(NW, h.shape[1], device=dev).index_add_(0, wi[tr], hc[tr]) / torch.bincount(wi[tr], minlength=NW)[:, None]
    r_cm = ranks(torch.nn.functional.normalize(hc[te], dim=1) @ torch.nn.functional.normalize(A, dim=1).T, y_te)
    Z = hc / scale
    vacc = {}
    for lam in LAMS:
        Wt, b = fit_probe(Z[fit_mask], wi[fit_mask], lam)
        vacc[lam] = float(((Z[val_mask] @ Wt.T + b).argmax(1) == wi[val_mask]).float().mean())
    lam = max(LAMS, key=lambda l: (vacc[l], l))
    Wt, b = fit_probe(Z[tr], wi[tr], lam)
    r_pr = ranks(Z[te] @ Wt.T + b, y_te)
    Mp = JPP[L].to(dev).float(); Mj = JL[L].to(dev).float()
    gU = g * Uf
    r_say = ranks(word_scores(h[te], gU @ Mp), y_te)
    r_sayj = ranks(word_scores(h[te], gU @ Mj), y_te)
    r_ll = ranks(word_scores(h[te], gU), y_te)
    s = (g * uX) @ Mp; sj = (g * uX) @ Mj
    fam = dict(SAY=s, SAYJ=sj, EMB=eX, UNEMB=g * uX)
    G = {"THINKa_vs_THINKb": geom(A, Wt)}
    for tn, T in (("THINKa", A), ("THINKb", Wt)):
        for fn, F in fam.items():
            G[f"{tn}_vs_{fn}"] = geom(T, F)
    G["SAY_vs_UNEMB"] = geom(s, g * uX)
    G["SAY_vs_SAYJ"] = geom(s, sj)
    ph, sh = r_pr <= 5, r_say <= 5
    ov = dict(probe_hits=int(ph.sum()), say_hits=int(sh.sum()), both=int((ph & sh).sum()),
              share_probe_hits_also_say=round(float((ph & sh).sum() / ph.sum().clamp(min=1)), 4),
              share_say_hits_also_probe=round(float((ph & sh).sum() / sh.sum().clamp(min=1)), 4))
    bykind = {}
    for k_i, kn in enumerate(KN):
        m = kind_of[y_te] == k_i
        bykind[kn] = dict(probe_top5=round(float((r_pr[m] <= 5).float().mean()), 3),
                          say_top5=round(float((r_say[m] <= 5).float().mean()), 3),
                          logit_top5=round(float((r_ll[m] <= 5).float().mean()), 3))
    on = A.norm(dim=1)
    res["layers"][L] = dict(probe=acc(r_pr), probe_lam=lam, probe_val=vacc, classmean=acc(r_cm), say_jpp=acc(r_say),
                            say_jlens=acc(r_sayj), logit_lens=acc(r_ll), geometry=G, overlap_top5=ov, by_kind=bykind,
                            offset_norm_median=round(float(on.median()), 3), resid_norm_median=round(float(h.norm(dim=1).median()), 3))
    if L in (8, 16, 24):
        causal[L] = dict(THINK=A.cpu(), SAY=s.cpu(), UNEMB=(g * uX).cpu(), typ=float(on.median()),
                         resid_med=float(h.norm(dim=1).median()))
    log(f"L{L:2d} probe {acc(r_pr)} (lam {lam}) cm {acc(r_cm)} say {acc(r_say)} sayJ {acc(r_sayj)} ll {acc(r_ll)} | "
        f"cos A-SAY c {G['THINKa_vs_SAY']['centred']} | ov {ov['share_probe_hits_also_say']}/{ov['share_say_hits_also_probe']} "
        f"| {time.time() - t0:.0f}s")
    json.dump(res, open(f"{BIG}/analysis.json", "w"), indent=1)
torch.save(causal, f"{BIG}/dirs.pt")
log("done")
