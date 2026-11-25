"""w12 train (Experiment 5B) [w12_train4 = w12_train3 with tau folded into the [B,d] side before the vocab product (one [B,V] buffer less);
w12_train3: same as w12_train.py (X on the GPU), val evaluation in chunks of --eval_bs, 4 CPU threads,
and configs whose map + fit json exist are skipped; identical training]: per readout layer, a [d,d] map M trained so that softmax over the WHOLE vocabulary of
tau * U norm(M h) matches a target built from the model's own output distributions (scripts/w12/w12_targets.py).
  loss = KL(target || softmax(tau * U norm(M h))) + mu * ||M - A||_F^2 / ||A||_F^2
  A (anchor = init): J (J-lens matrix, form i), JPP (J++ matrix as the evaluator loads it, form ii: J++ + Delta with
  Delta = M - JPP init 0), I (identity = tuned lens from scratch, linear, no bias; form iii, mu = 0).
  tau: one learned scalar temperature per map (init by a 5-point grid); it does not change any rank.
  Adam, lr = lr_rel * rms(J_l) on M for every anchor (lr_rel 3e-3: best val KL of 3e-4/1e-3/3e-3 on L12 T1), batch 1024, eval on val every 250 steps, patience 4,
  max --max_steps; best val (the KL part only) is kept. --shuf: targets permuted across all samples (seed 0).
Config names: <target>_<anchor>_mu<mu>[_shuf]  e.g. T4_J_mu0, T1_JPP_mu1, T16_I_mu0, T1_J_mu0_shuf
Output: .hf27/w12/<tag>/maps/<cfg>.pt  {layer: {"M": fp16 [d,d], "tau": float}} (one file per config, all layers
merged by w12_merge), here per layer: .hf27/w12/<tag>/maps/<cfg>_L<l>.pt ; runs/w12/fits/<tag>_<cfg>_L<l>.json
Usage: python w12_train.py <model> --layer L --cfgs a,b,c [--max_steps 4000] [--lr_rel 1e-3]"""
import argparse
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--cfgs", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--max_steps", type=int, default=4000)
ap.add_argument("--lr_rel", type=float, default=3e-3)
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--ntrain", type=int, default=None, help="use only the first n train samples (smoke)")
ap.add_argument("--suffix", default="")
ap.add_argument("--nosave", action="store_true")
ap.add_argument("--eval_bs", type=int, default=1024)
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
M_ = args.model
TAG = "2b" if "2B" in M_ else "9b"
D = args.data or f"{R}/.hf27/w12/{TAG}"
OUT = f"{D}/maps"
ROUT = f"{R}/runs/w12/fits"
os.makedirs(OUT, exist_ok=True); os.makedirs(ROUT, exist_ok=True)
sys.path.insert(0, f"{R}/scripts/perf1"); sys.path.insert(0, f"{R}/src")
from lite_model import get_lite_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

dev = "cuda"
EVAL_PATHS = {"2b": ("artifacts/baselines/jlens_prefit.pt", "artifacts/baselines/jpp_prefit.pt"),
              "9b": ("runs/v6_9b/jlens9b.pt", "artifacts/baselines/jpp9b_prefit.pt")}[TAG]
l = args.layer
t_all = time.time()
model = get_lite_model(M_)
U = model._lm_head.weight.float()  # [V, d]
norm = model._final_norm
eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
with torch.no_grad():
    gamma = norm(torch.ones(1, U.shape[1], device=dev, dtype=next(norm.parameters()).dtype)).float()[0] * (1 + eps) ** .5
d = U.shape[1]
anchors = {}
for nm, p in zip(("J", "JPP"), EVAL_PATHS):
    anchors[nm] = load_lens_file(f"{R}/{p}", hf_model_name=M_).jacobians_L_dict_FN[l].to(dev).float()
anchors["I"] = torch.eye(d, device=dev)

meta = torch.load(f"{D}/meta.pt", map_location="cpu")
val = meta["val"]
tr_i = torch.nonzero(~val)[:, 0]
va_i = torch.nonzero(val)[:, 0]
if args.ntrain:
    tr_i = tr_i[:args.ntrain]
del meta
X = torch.load(f"{D}/X_L{l}.pt").to(dev)
torch.set_num_threads(4)
TG = torch.load(f"{D}/targets.pt")
print(f"[w12 train] {TAG} L{l}: X {tuple(X.shape)}, train {len(tr_i)}, val {len(va_i)}, load {time.time() - t_all:.0f}s",
      flush=True)
tr_i, va_i = tr_i.to(dev), va_i.to(dev)
g = torch.Generator(device=dev).manual_seed(123)
sub_tr = tr_i[torch.randperm(len(tr_i), device=dev, generator=g)[:len(va_i)]]


def logits(h, Mt, logt):
    v = h.float() @ Mt.T
    z = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * gamma
    return (logt.exp() * z) @ U.T


def kl(h, Mt, logt, I, P, Hp):
    lg = logits(h, Mt, logt)
    lq = lg.gather(1, I) - torch.logsumexp(lg, -1, keepdim=True)
    return (Hp - (P * lq).sum(-1)).mean()


@torch.no_grad()
def xrows(r):
    return X[r]


def kl_rows(rows, Mt, logt, T, bs=None):
    bs = bs or args.eval_bs
    I, P, Hp, ok = T
    rows = rows[ok[rows]]
    s = 0.0
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        s += kl(xrows(r), Mt, logt, I[r], P[r], Hp[r]).item() * len(r)
    return s / len(rows)


def tgt(name, shuf):
    ids, p = TG[name]
    I, P = ids.long().to(dev), p.float().to(dev)
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)  # all-junk rows: P = 0 -> dropped below
    if shuf:
        perm = torch.randperm(len(I), generator=torch.Generator().manual_seed(0)).to(dev)
        I, P = I[perm], P[perm]
    Hp = torch.where(P > 0, P * P.clamp_min(1e-30).log(), torch.zeros_like(P)).sum(-1)
    return I, P, Hp, P.sum(-1) > 0


common = {k: tgt(k, False) for k in ("T1", "T16")}
for cfg in args.cfgs.split(","):
    t0 = time.time()
    if os.path.exists(f"{OUT}/{cfg}{args.suffix}_L{l}.pt") and os.path.exists(f"{ROUT}/{TAG}_{cfg}{args.suffix}_L{l}.json"):
        print(f"[{TAG} L{l} {cfg}] exists, skipped", flush=True)
        continue
    parts = cfg.split("_")
    tname, anc, mu = parts[0], parts[1], float(parts[2][2:])
    shuf = len(parts) > 3 and parts[3] == "shuf"
    T = tgt(tname, shuf)
    A = anchors[anc]
    An2 = A.pow(2).sum()
    Mt = A.clone().requires_grad_(True)
    logt = torch.zeros((), device=dev)
    with torch.no_grad():
        grid = [(kl_rows(sub_tr[:4096], Mt, torch.tensor(a, device=dev), T), a) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        logt.fill_(min(grid)[1])
    logt.requires_grad_(True)
    lr = args.lr_rel * anchors["J"].pow(2).mean().sqrt().item()  # same absolute lr for every anchor
    opt = torch.optim.Adam([{"params": [Mt], "lr": lr}, {"params": [logt], "lr": 1e-2}])
    with torch.no_grad():
        v0, tr0 = kl_rows(va_i, Mt, logt, T), kl_rows(sub_tr, Mt, logt, T)
    best = (v0, 0, Mt.detach().clone(), logt.detach().clone())
    step, bad, hist = 0, 0, []
    I_, P_, H_, ok_ = T
    tr_ok = tr_i[ok_[tr_i]]
    while step < args.max_steps and bad < args.patience:
        perm = tr_ok[torch.randperm(len(tr_ok), device=dev, generator=g)]
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            r = perm[i:i + args.bs]
            loss = kl(xrows(r), Mt, logt, I_[r], P_[r], H_[r])
            if mu > 0:
                loss = loss + mu * (Mt - A).pow(2).sum() / An2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % args.eval_every == 0:
                vk = kl_rows(va_i, Mt.detach(), logt.detach(), T)
                hist.append((step, round(vk, 5)))
                if vk < best[0] - 1e-4:
                    best, bad = (vk, step, Mt.detach().clone(), logt.detach().clone()), 0
                else:
                    bad += 1
                if bad >= args.patience or step >= args.max_steps:
                    break
    _, bstep, Mb, lb = best
    with torch.no_grad():
        rep = dict(cfg=cfg, layer=l, target=tname, anchor=anc, mu=mu, shuf=shuf, lr=lr, tau=float(lb.exp()),
                   val0=v0, train0=tr0, val=best[0], train=kl_rows(sub_tr, Mb, lb, T), step=bstep, stopped=step,
                   rel_dist=((Mb - A).norm() / A.norm()).item(),
                   rel_dist_J=((Mb - anchors["J"]).norm() / anchors["J"].norm()).item(),
                   val_T1=kl_rows(va_i, Mb, lb, common["T1"]), val_T16=kl_rows(va_i, Mb, lb, common["T16"]),
                   seconds=time.time() - t0, hist=hist)
    print(f"[{TAG} L{l} {cfg}] val KL {v0:.4f} -> {best[0]:.4f} (train {tr0:.4f} -> {rep['train']:.4f}) step "
          f"{bstep}/{step} |M-A|/|A| {rep['rel_dist']:.3f} tau {rep['tau']:.2f} ({rep['seconds']:.0f}s)", flush=True)
    if not args.nosave:
        torch.save({"M": Mb.half().cpu(), "tau": float(lb.exp())}, f"{OUT}/{cfg}{args.suffix}_L{l}.pt")
    json.dump(rep, open(f"{ROUT}/{TAG}_{cfg}{args.suffix}_L{l}.json", "w"), indent=1)
    del Mt, Mb, opt, T
    torch.cuda.empty_cache()
print(f"[w12 train] done {time.time() - t_all:.0f}s", flush=True)
