"""w9 fit: linear and one-hidden-layer MLP A-lens maps, L2 (ridge target) and cross-entropy losses, two train sizes.
For one model x question x train size (x shuffled control), per readout layer, in dependency order:
  lin_l2   closed-form ridge exactly as scripts/ask3/ask3_fit9.py (target T = mean of kept unit answer states, junk
           rule on the unmasked top-1; centred X; penalty c from GRID x mean eigenvalue chosen on val by top-10
           agreement with the elicited word).
  lin_ce   scripts/w2/w2_ce_fit.py verbatim recipe (KL(p || softmax over the full vocabulary of tau*unembed(unit(.)))),
           target p from w9_targets (--target, default w2 = w2's build_p, K=10); init lin_l2; Adam lr 3e-3 x std(init);
           batch 256; eval every 50 steps; patience 6; max 3000 steps; early stop on val KL.
  mlp<m>_l2 map(h) = E h + b + W2 relu(W1 z + c1), z = (h - mx) / sx (sx = per-dim std of train X; a fixed affine
           rescaling absorbed in W1, c1); W2 = 0 at init so step 0 == lin_l2; E, b init lin_l2. AdamW: E, b lr 3e-3 x
           std(init) (no decay) ONLY with --mlp_train_lin (default: E, b frozen at the linear solution -- training them
           jointly only continued the linear map's overfitting in the probe), W1, c1, W2 lr --lr_mlp with weight decay
           --wd on W1, W2. MSE to T; early stop on val
           MSE (eval every 50, patience 8, max 5000 steps).
  mlp<m>_ce same MLP, init lin_ce (incl. tau), KL loss as lin_ce, early stop on val KL (patience 8, max 5000).
Split: val 0..999, test 1000..1999 (never touched), train 2000..2000+n (--n 28500 = the old train set; --n all).
--shuf: answers (T and p) permuted across the valid train+val passages (seed 0) BEFORE everything; the whole pipeline
then runs unchanged (c chosen and early stopping on the permuted val answers).
Output: .hf27/w9/maps/E_<tag>_<q>_<arch>_<loss>_<n>[_shuf][_t<target>].pt (alens-loadable: mx, W/scale, my; MLP adds
W1, c1, W2, sx) and runs/w9/fits/<same stem>.json (per layer: train/val loss at init and best, steps, c, a10).
Usage: python w9_fit.py <model> <topic|spec> --n 28500|all [--shuf] [--target w2] [--archs ...] [--smoke]
"""
import argparse
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.path.insert(0, f"{R}/scripts/perf1")
sys.path.insert(0, f"{R}/scripts/w9")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("question", choices=["topic", "spec"])
ap.add_argument("--n", default="28500")
ap.add_argument("--shuf", action="store_true")
ap.add_argument("--target", default="w2")
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--archs", default="lin,mlp512,mlp2048")
ap.add_argument("--losses", default="l2,ce")
ap.add_argument("--lr_mlp", type=float, default=1e-4)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--mlp_train_lin", action="store_true", help="also train E, b in the MLP fits (default: frozen)")
ap.add_argument("--layers", default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--seed", type=int, default=0, help="MLP init seed offset (W1, c1); 0 = the main runs")
ap.add_argument("--tag_extra", default="")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
M = args.model
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
if args.layers:
    LAYERS = [int(x) for x in args.layers.split(",")]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
QD = {"topic": ("ask3", "t"), "spec": ("ask5", "s")}[args.question]
YFILE = f"{R}/.hf27/{QD[0]}/acts_{TAG}.pt"
NEWDIR = f"{R}/.hf27/w9/{TAG}_new"
NOLD, NV, NT = 30500, 1000, 1000
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
BS, EVAL_EVERY, LR_REL = 256, 50, 3e-3
PAT = {"lin": 6, "mlp": 8}
MAXS = {"lin": 3000, "mlp": 5000}
if args.smoke:
    MAXS = {"lin": 300, "mlp": 600}
dev = "cuda"
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402
import w9_targets  # noqa: E402

model = get_lite_model(M)
C.setup(model)
Wu, V, d, gamma = C.Wu, C.V, C.d, C.gamma
unit = C.unit
OUT = f"{R}/.hf27/w9/maps" + ("/smoke" if args.smoke else "")
ROUT = f"{R}/runs/w9/fits" + ("/smoke" if args.smoke else "")
os.makedirs(OUT, exist_ok=True); os.makedirs(ROUT, exist_ok=True)
t_all = time.time()

# ---------------------------------------------------------------- data (old 30,500 + new rows if needed)
use_new = args.n == "all"
Yo = torch.load(YFILE, map_location="cpu")["Y_new"]
if use_new:
    Ynew = torch.stack([torch.load(f"{NEWDIR}/Y{QD[1]}_w{k}.pt") for k in range(3)])
    Y = torch.cat([Yo, Ynew], 1); del Ynew
else:
    Y = Yo
del Yo
N = Y.shape[1]
NTR = N - NV - NT if use_new else int(args.n)
assert NV + NT + NTR <= N
NUSE = NV + NT + NTR
Y = Y[:, :NUSE]
print(f"[w9 fit] {M} {args.question} n={args.n} (train rows {NV + NT}..{NUSE - 1}) shuf={args.shuf} target={args.target} "
      f"archs={args.archs} losses={args.losses}; Y {tuple(Y.shape)}", flush=True)
with torch.no_grad():
    Yk = [unit(Y[k].float()) for k in range(3)]
    del Y
    raw1 = [C.topk(y, 1, m=None)[:, 0] for y in Yk]
    junk = torch.stack([C.junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    T_raw = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
    valid = (cnt > 0).cpu()
    word = C.topk(T_raw, 10)[:, 0]
    wk = keepk.float() / cnt.clamp_min(1).float()
    ids, val = w9_targets.TARGETS[args.target](C, Yk, wk, args.K, 1.0)
    del Yk
print(f"[w9 fit] wordings dropped {int(junk.sum())}/{3 * NUSE}; passages dropped {int((~valid).sum())}; targets built "
      f"{time.time() - t_all:.0f}s", flush=True)
tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
if args.shuf:
    pool = torch.nonzero(tr_m | va_m)[:, 0]
    perm = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))]
    T_raw[pool] = T_raw[perm]; ids[pool] = ids[perm]; val[pool] = val[perm]; word[pool.to(dev)] = word[perm.to(dev)]
tr_i, va_i = (torch.nonzero(m)[:, 0].to(dev) for m in (tr_m, va_m))
sub_i = tr_i[torch.randperm(len(tr_i), generator=torch.Generator().manual_seed(1))[:4000].to(dev)]
Tt = T_raw.to(dev)
del T_raw
Ik, Pv = ids.long().to(dev), val.to(dev)
Hp = (torch.where(Pv > 0, Pv * Pv.clamp_min(1e-30).log(), torch.zeros_like(Pv))).sum(-1)
del ids, val
print(f"[w9 fit] train {len(tr_i)}, val {len(va_i)}", flush=True)
Xo = torch.load(XFILE, map_location="cpu")["X"]
Xn = torch.load(f"{NEWDIR}/X.pt") if use_new else None
print(f"[w9 fit] X loaded {time.time() - t_all:.0f}s", flush=True)
Wk = Wu.float()


class Ridge:  # verbatim from ask3_fit9.py
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
    x = X[rows] - P["mx"]
    out = x @ P["W"] + P["b"]
    if "W1" in P:
        out = out + torch.relu((x / P["sx"]) @ P["W1"].T + P["c1"]) @ P["W2"].T
    return out


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


def train(X, P0, kind, fam, l):
    """P0: dict of init tensors. Trains all of them; returns best params (by val loss) and a report."""
    P = {k: v.clone().requires_grad_(k not in ("mx", "sx")) for k, v in P0.items()}
    groups = []
    if fam == "lin" or args.mlp_train_lin:
        groups += [{"params": [P["W"]], "lr": LR_REL * P0["W"].std().item(), "weight_decay": 0.0},
                   {"params": [P["b"]], "lr": LR_REL * P0["b"].std().item(), "weight_decay": 0.0}]
    else:  # MLP: the linear part stays at its fitted solution; only the hidden branch (and tau) is trained
        P["W"].requires_grad_(False); P["b"].requires_grad_(False)
    if kind == "ce":
        groups.append({"params": [P["logt"]], "lr": 1e-2, "weight_decay": 0.0})
    if fam == "mlp":
        groups += [{"params": [P["W1"], P["W2"]], "lr": args.lr_mlp, "weight_decay": args.wd},
                   {"params": [P["c1"]], "lr": args.lr_mlp, "weight_decay": 0.0}]
    opt = torch.optim.Adam(groups) if fam == "lin" else torch.optim.AdamW(groups)
    with torch.no_grad():
        if kind == "ce" and fam == "lin":  # w2: initial tau from a 5-point grid on 2000 train rows
            g = [(loss_fn(X, tr_i[:2000], {**P, "logt": torch.tensor(float(a), device=dev)}, kind).item(), a)
                 for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
            P["logt"].fill_(min(g)[1])
        v0 = loss_fn(X, va_i, P, kind).item()
        tr0 = loss_fn(X, sub_i, P, kind).item()
        best = (v0, 0, {k: v.detach().clone() for k, v in P.items()})
    gen = torch.Generator(device=dev).manual_seed(l)
    step, bad, t0 = 0, 0, time.time()
    while step < MAXS[fam] and bad < PAT[fam]:
        perm = tr_i[torch.randperm(len(tr_i), device=dev, generator=gen)]
        for i in range(0, len(perm) - BS + 1, BS):
            loss = loss_fn(X, perm[i:i + BS], P, kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                with torch.no_grad():
                    vk = loss_fn(X, va_i, P, kind).item()
                if args.verbose:
                    print(f"      {fam}-{kind} step {step}: batch {loss.item():.4f} val {vk:.4f}", flush=True)
                if vk < best[0] - 1e-4 * (1 if kind == "ce" else 0.01):
                    best, bad = (vk, step, {k: v.detach().clone() for k, v in P.items()}), 0
                else:
                    bad += 1
                if bad >= PAT[fam] or step >= MAXS[fam]:
                    break
    with torch.no_grad():
        Pb = best[2]
        rep = dict(val0=v0, val=best[0], train0=tr0,
                   train=loss_fn(X, sub_i, Pb, kind).item(), step=best[1], stopped=step, seconds=time.time() - t0,
                   a10=a10(X, Pb, va_i))
    return Pb, rep


def pack(P, c=None):
    sc = P["W"].abs().max().item() or 1.0
    m = dict(mx=P["mx"].cpu(), W=(P["W"] / sc).half().cpu(), scale=sc, my=P["b"].detach().cpu(), c=c)
    if "logt" in P:
        m["tau"] = float(P["logt"])
    if "W1" in P:
        m.update(W1=P["W1"].detach().half().cpu(), c1=P["c1"].detach().cpu(), W2=P["W2"].detach().half().cpu(),
                 sx=P["sx"].cpu())
    return m


archs, losses = args.archs.split(","), args.losses.split(",")
nm = "98k" if use_new else ("28k" if args.n == "28500" else args.n)
stem = lambda a, lo: f"E_{TAG}_{args.question}_{a}_{lo}_{nm}" + ("_shuf" if args.shuf else "") + \
    (f"_t{args.target}" if args.target != "w2" and lo == "ce" else "") + args.tag_extra  # noqa: E731
maps = {(a, lo): {} for a in archs for lo in losses}
reps = {(a, lo): {} for a in archs for lo in losses}
for l in LAYERS:
    X = (torch.cat([Xo[l][:NOLD], Xn[l]]) if use_new else Xo[l])[:NUSE].float().to(dev)
    t0 = time.time()
    r = Ridge(X[tr_i])
    r.setY(Tt[tr_i])
    mx = r.mx
    sc = []
    for c_ in GRID:
        Pc = dict(mx=mx, W=r.W(c_), b=r.my)
        sc.append(a10(X, Pc, va_i))
    bi = max(range(len(GRID)), key=lambda i: (sc[i], i))
    c = GRID[bi]
    L2 = dict(mx=mx, W=r.W(c), b=r.my.clone())
    del r
    torch.cuda.empty_cache()
    with torch.no_grad():
        rep = dict(c=c, a10_by_c=sc, val=loss_fn(X, va_i, L2, "l2").item(), train=loss_fn(X, sub_i, L2, "l2").item(),
                   a10=sc[bi], seconds=time.time() - t0)
    if "lin" in archs and "l2" in losses:
        maps[("lin", "l2")][l] = pack(L2, c); reps[("lin", "l2")][l] = rep
    print(f"[{TAG} {args.question} {nm}] L{l} lin_l2: c={c:g} val MSE {rep['val']:.4f} train {rep['train']:.4f} "
          f"a10 {rep['a10']:.3f} ({rep['seconds']:.0f}s)", flush=True)
    CE = None
    if "ce" in losses:
        CE, rep = train(X, {**L2, "logt": torch.zeros((), device=dev)}, "ce", "lin", l)
        if "lin" in archs:
            maps[("lin", "ce")][l] = pack(CE); reps[("lin", "ce")][l] = rep
        print(f"[{TAG} {args.question} {nm}] L{l} lin_ce: val KL {rep['val0']:.3f} -> {rep['val']:.3f} (train "
              f"{rep['train0']:.3f} -> {rep['train']:.3f}) step {rep['step']}/{rep['stopped']} a10 {rep['a10']:.3f} "
              f"({rep['seconds']:.0f}s)", flush=True)
    sx = X[tr_i].std(0).clamp_min(1e-3)
    for a in archs:
        if not a.startswith("mlp"):
            continue
        m = int(a[3:])
        g = torch.Generator(device=dev).manual_seed(1000 + l + 100000 * args.seed)
        W1 = (torch.rand(m, d, device=dev, generator=g) * 2 - 1) / d ** 0.5  # nn.Linear-style uniform init
        c1 = (torch.rand(m, device=dev, generator=g) * 2 - 1) / d ** 0.5
        for lo in losses:
            base = L2 if lo == "l2" else CE
            P0 = {k: v.detach() for k, v in base.items()}
            P0.update(W1=W1, c1=c1, W2=torch.zeros(d, m, device=dev), sx=sx)
            Pb, rep = train(X, P0, lo, "mlp", l)
            maps[(a, lo)][l] = pack(Pb, c if lo == "l2" else None); reps[(a, lo)][l] = rep
            print(f"[{TAG} {args.question} {nm}] L{l} {a}_{lo}: val {rep['val0']:.4f} -> {rep['val']:.4f} (train "
                  f"{rep['train0']:.4f} -> {rep['train']:.4f}) step {rep['step']}/{rep['stopped']} a10 {rep['a10']:.3f} "
                  f"({rep['seconds']:.0f}s)", flush=True)
            del Pb
    del X
    torch.cuda.empty_cache()
for key, mp in maps.items():
    if not mp:
        continue
    s = stem(*key)
    torch.save(mp, f"{OUT}/{s}.pt")
    json.dump(dict(args=vars(args), layers={str(k): v for k, v in reps[key].items()}), open(f"{ROUT}/{s}.json", "w"), indent=1)
    print(f"[w9 fit] saved {OUT}/{s}.pt", flush=True)
print(f"[w9 fit] done {time.time() - t_all:.0f}s", flush=True)
