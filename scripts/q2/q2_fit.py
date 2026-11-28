"""q2 fit: linear A-lens maps for the new questions (q3..q7) with the w9 recipe (scripts/w9/w9_fit.py, archs "lin",
losses l2 + ce, n = all = 98,000 train passages), plus the answer-distribution report.
Per readout layer: lin_l2 = closed-form ridge (penalty c from w9's GRID x mean eigenvalue, chosen on val by top-10
agreement with the elicited word), then lin_ce = KL(p || softmax(tau * unembed(unit(map)))) from the lin_l2 init with
Adam (lr 3e-3 x std(init)), batch 256, eval every 50 steps, patience 6, max 3000 steps, early stop on val KL.
Target p: w9_targets "w2" rule (per wording: top-10 word tokens with junk/non-word tokens removed, softmax, averaged over
the kept wordings), generalised from 3 wordings to whichever wordings are given (--wordings).
Split as w9: val rows 0..999, test 1000..1999 (never touched), train 2000..99999.
Data: X = .hf27/ask2/acts_9b.pt | .hf27/pod_root/e1/acts_2b.pt (rows 0..30499) + .hf27/w9/<tag>_new/X.pt;
      Y = .hf27/q2/<tag>/Y_<q>_w<k>.pt (scripts/q2/q2_collect.py).
Also --dist_only for "topic" / "spec" (old Y files, rows 2000..3999) to compare answer distributions.
Output: .hf27/q2/maps/E_<tag>_<q>_lin_{l2,ce}_98k<_wK>.pt (w9 map format) + runs/q2/fits/<stem>.json,
        runs/q2/answers_<tag>_<q><_wK>.json (answer distribution over train rows 2000..3999).
Usage: python q2_fit.py <model> <q3|q4|q5|q6|q7|topic|spec> [--wordings 0,1,2] [--dist_only]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.path.insert(0, f"{R}/scripts/perf1")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("question")
ap.add_argument("--wordings", default="0,1,2")
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--dist_only", action="store_true")
ap.add_argument("--layers", default=None)
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
M = args.model
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
if args.layers:
    LAYERS = [int(x) for x in args.layers.split(",")]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
NEWDIR = f"{R}/.hf27/w9/{TAG}_new"
NOLD, NV, NT = 30500, 1000, 1000
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
BS, EVAL_EVERY, LR_REL, PAT, MAXS = 256, 50, 3e-3, 6, 3000
dev = "cuda"
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

model = get_lite_model(M)
C.setup(model)
Wu, V, d, gamma = C.Wu, C.V, C.d, C.gamma
unit = C.unit
WK = [int(x) for x in args.wordings.split(",")]
sfx = "" if WK == [0, 1, 2] else "_w" + "".join(map(str, WK))
OUT, ROUT = f"{R}/.hf27/q2/maps", f"{R}/runs/q2/fits"
os.makedirs(OUT, exist_ok=True); os.makedirs(ROUT, exist_ok=True)
t_all = time.time()

# ------------------------------------------------------------------ Y
if args.question in ("topic", "spec"):
    assert args.dist_only
    d_ = {"topic": "ask3", "spec": "ask5"}[args.question]
    Y = torch.load(f"{R}/.hf27/{d_}/acts_{TAG}.pt", map_location="cpu")["Y_new"][WK, :4000]
else:
    Y = torch.stack([torch.load(f"{R}/.hf27/q2/{TAG}/Y_{args.question}_w{k}.pt") for k in WK])
    if args.dist_only:
        Y = Y[:, :4000]
N = Y.shape[1]
nw = len(WK)
print(f"[q2 fit] {M} {args.question} wordings {WK}: Y {tuple(Y.shape)}", flush=True)


@torch.no_grad()
def build_targets(Yk, wk, K, rows=None, bs=250):
    """w9_targets.build_w2 for any number of wordings."""
    keep = ~C.nonword & ~C.junk_tok
    rows = torch.arange(Yk[0].shape[0]) if rows is None else rows
    ids_all, val_all = [], []
    for s0 in range(0, len(rows), bs):
        r = rows[s0:s0 + bs]
        D = torch.zeros(len(r), C.V, device=dev)
        for k in range(len(Yk)):
            vv, ii = C.logits(Yk[k][r]).masked_fill(~keep, float("-inf")).topk(K, -1)
            D.scatter_add_(1, ii, torch.softmax(vv, -1) * wk[k, r.to(dev), None])
        vv, ii = D.topk(3 * K, -1)
        ids_all.append(ii.int().cpu()); val_all.append(vv.cpu())
    return torch.cat(ids_all), torch.cat(val_all)


with torch.no_grad():
    Yk = [unit(Y[k].float()) for k in range(nw)]
    del Y
    raw1 = [C.topk(y, 1, m=None)[:, 0] for y in Yk]
    junk = torch.stack([C.junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    valid = (cnt > 0).cpu()
    wk = keepk.float() / cnt.clamp_min(1).float()
    # ---------------- answer distribution over train rows 2000..3999
    dr = torch.arange(2000, 4000)
    ids_d, val_d = build_targets(Yk, wk, args.K, rows=dr)
    top1 = Counter(C.strs[int(i)] for i in ids_d[:, 0])
    mass = torch.zeros(C.V)
    mass.scatter_add_(0, ids_d.long().flatten(), val_d.flatten())
    mass /= len(dr)
    mt = mass.topk(30)
    raw_top1 = [Counter(C.strs[int(i)] for i in raw1[k][dr.to(dev)]) for k in range(nw)]
    rep = dict(model=M, question=args.question, wordings=WK, rows=[2000, 3999],
               junk_top1_rate=[float(junk[k, dr.to(dev)].float().mean()) for k in range(nw)],
               top1_answers=[(s, c / len(dr)) for s, c in top1.most_common(30)],
               top_mass=[(C.strs[int(i)], float(v)) for v, i in zip(mt.values, mt.indices)],
               distinct_top1=len(top1), top1_share_max=top1.most_common(1)[0][1] / len(dr),
               top10_top1_share=sum(c for _, c in top1.most_common(10)) / len(dr),
               raw_top1_per_wording=[[(s, c / len(dr)) for s, c in rc.most_common(8)] for rc in raw_top1])
    json.dump(rep, open(f"{R}/runs/q2/answers_{TAG}_{args.question}{sfx}.json", "w"), ensure_ascii=False, indent=1)
    print(f"[q2 fit] answers {args.question}: distinct top-1 {len(top1)}/2000, max share {rep['top1_share_max']:.3f}, "
          f"top-10 answers cover {rep['top10_top1_share']:.3f}; junk top-1 per wording {rep['junk_top1_rate']}",
          flush=True)
    print("   top-1:", ", ".join(f"{s!r} {v:.3f}" for s, v in rep["top1_answers"]), flush=True)
    print("   mass :", ", ".join(f"{s!r} {v:.3f}" for s, v in rep["top_mass"]), flush=True)
    for k in range(nw):
        print(f"   raw top-1 w{WK[k]}:", ", ".join(f"{s!r} {v:.3f}" for s, v in rep["raw_top1_per_wording"][k]), flush=True)
if args.dist_only:
    sys.exit(0)

with torch.no_grad():
    T_raw = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(nw)) / cnt.cpu().clamp_min(1)[:, None].float()
    word = C.topk(T_raw, 10)[:, 0]
    ids, val = build_targets(Yk, wk, args.K)
    del Yk
print(f"[q2 fit] wordings dropped {int(junk.sum())}/{nw * N}; passages dropped {int((~valid).sum())}; targets "
      f"{time.time() - t_all:.0f}s", flush=True)
tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
tr_i, va_i = (torch.nonzero(m)[:, 0].to(dev) for m in (tr_m, va_m))
sub_i = tr_i[torch.randperm(len(tr_i), generator=torch.Generator().manual_seed(1))[:4000].to(dev)]
Tt = T_raw.to(dev)
del T_raw
Ik, Pv = ids.long().to(dev), val.to(dev)
Hp = (torch.where(Pv > 0, Pv * Pv.clamp_min(1e-30).log(), torch.zeros_like(Pv))).sum(-1)
del ids, val
print(f"[q2 fit] train {len(tr_i)}, val {len(va_i)}", flush=True)
Xo = torch.load(XFILE, map_location="cpu")["X"]
Xn = torch.load(f"{NEWDIR}/X.pt")
print(f"[q2 fit] X loaded {time.time() - t_all:.0f}s", flush=True)
Wk = Wu.float()


class Ridge:  # = w9_fit.Ridge (ask3_fit9.py)
    def __init__(self, X):
        self.mx = X.mean(0)
        self.Xc = (X - self.mx).double()
        s, U = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.s, self.U = s.clamp_min(0), U
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.UtB = self.U.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.U @ (self.UtB / (self.s + c * self.scale)[:, None])).float()


def fwd(X, rows, P):
    return (X[rows] - P["mx"]) @ P["W"] + P["b"]


def loss_fn(X, rows, P, kind):
    if len(rows) > BS and not torch.is_grad_enabled():
        return sum(loss_fn(X, rows[i:i + BS], P, kind) * len(rows[i:i + BS]) for i in range(0, len(rows), BS)) / len(rows)
    h = fwd(X, rows, P)
    if kind == "l2":
        return (h - Tt[rows]).pow(2).mean()
    lg = P["logt"].exp() * (unit(h) * gamma) @ Wk.T
    lq = lg.gather(1, Ik[rows]) - torch.logsumexp(lg, -1, keepdim=True)
    return (Hp[rows] - (Pv[rows] * lq).sum(-1)).mean()


@torch.no_grad()
def a10(X, P, rows):
    return C.agree(C.topk(torch.cat([fwd(X, rows[i:i + BS], P) for i in range(0, len(rows), BS)])), word[rows]).float().mean().item()


def train(X, P0, l):  # = w9_fit.train for fam "lin", kind "ce"
    P = {k: v.clone().requires_grad_(k != "mx") for k, v in P0.items()}
    opt = torch.optim.Adam([{"params": [P["W"]], "lr": LR_REL * P0["W"].std().item()},
                            {"params": [P["b"]], "lr": LR_REL * P0["b"].std().item()},
                            {"params": [P["logt"]], "lr": 1e-2}])
    with torch.no_grad():
        g = [(loss_fn(X, tr_i[:2000], {**P, "logt": torch.tensor(float(a), device=dev)}, "ce").item(), a)
             for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        P["logt"].fill_(min(g)[1])
        v0 = loss_fn(X, va_i, P, "ce").item()
        tr0 = loss_fn(X, sub_i, P, "ce").item()
        best = (v0, 0, {k: v.detach().clone() for k, v in P.items()})
    gen = torch.Generator(device=dev).manual_seed(l)
    step, bad, t0 = 0, 0, time.time()
    while step < MAXS and bad < PAT:
        perm = tr_i[torch.randperm(len(tr_i), device=dev, generator=gen)]
        for i in range(0, len(perm) - BS + 1, BS):
            loss = loss_fn(X, perm[i:i + BS], P, "ce")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    vk = loss_fn(X, va_i, P, "ce").item()
                if vk < best[0] - 1e-4:
                    best, bad = (vk, step, {k: v.detach().clone() for k, v in P.items()}), 0
                else:
                    bad += 1
                if bad >= PAT or step >= MAXS:
                    break
    with torch.no_grad():
        Pb = best[2]
        rep = dict(val0=v0, val=best[0], train0=tr0, train=loss_fn(X, sub_i, Pb, "ce").item(), step=best[1],
                   stopped=step, seconds=time.time() - t0, a10=a10(X, Pb, va_i), tau=float(Pb["logt"].exp()))
    return Pb, rep


def pack(P, c=None):
    sc = P["W"].abs().max().item() or 1.0
    m = dict(mx=P["mx"].cpu(), W=(P["W"] / sc).half().cpu(), scale=sc, my=P["b"].detach().cpu(), c=c)
    if "logt" in P:
        m["tau"] = float(P["logt"])
    return m


maps = {"l2": {}, "ce": {}}
reps = {"l2": {}, "ce": {}}
for l in LAYERS:
    X = torch.cat([Xo[l][:NOLD], Xn[l]])[:N].float().to(dev)
    t0 = time.time()
    r = Ridge(X[tr_i])
    r.setY(Tt[tr_i])
    sc = [a10(X, dict(mx=r.mx, W=r.W(c_), b=r.my), va_i) for c_ in GRID]
    bi = max(range(len(GRID)), key=lambda i: (sc[i], i))
    c = GRID[bi]
    L2 = dict(mx=r.mx, W=r.W(c), b=r.my.clone())
    del r
    torch.cuda.empty_cache()
    with torch.no_grad():
        rep = dict(c=c, a10_by_c=sc, val=loss_fn(X, va_i, L2, "l2").item(), train=loss_fn(X, sub_i, L2, "l2").item(),
                   a10=sc[bi], seconds=time.time() - t0)
    maps["l2"][l] = pack(L2, c); reps["l2"][l] = rep
    print(f"[{TAG} {args.question}] L{l} lin_l2: c={c:g} val MSE {rep['val']:.4f} a10 {rep['a10']:.3f} "
          f"({rep['seconds']:.0f}s)", flush=True)
    CE, rep = train(X, {**L2, "logt": torch.zeros((), device=dev)}, l)
    maps["ce"][l] = pack(CE); reps["ce"][l] = rep
    print(f"[{TAG} {args.question}] L{l} lin_ce: val KL {rep['val0']:.3f} -> {rep['val']:.3f} (train {rep['train0']:.3f} "
          f"-> {rep['train']:.3f}) step {rep['step']}/{rep['stopped']} a10 {rep['a10']:.3f} ({rep['seconds']:.0f}s)",
          flush=True)
    del X, CE
    torch.cuda.empty_cache()
for lo in ("l2", "ce"):
    s = f"E_{TAG}_{args.question}_lin_{lo}_98k{sfx}"
    torch.save(maps[lo], f"{OUT}/{s}.pt")
    json.dump(dict(args=vars(args), layers={str(k): v for k, v in reps[lo].items()}), open(f"{ROUT}/{s}.json", "w"),
              indent=1)
    print(f"[q2 fit] saved {OUT}/{s}.pt", flush=True)
print(f"[q2 fit] done {time.time() - t_all:.0f}s", flush=True)
