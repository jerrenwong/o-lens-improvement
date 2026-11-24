"""W2a: cross-entropy ("P3v") ask-map for either question, parameterised copy of scripts/st1/st1_fit.py (target p) +
scripts/st1/st1_p3v.py (fit). Nothing in the recipe changes except the inputs:
  question  topic -> answer states .hf27/ask3/acts_<tag>.pt["Y_new"], ridge init .hf27/ask3/E_<tag>_t3(_shuf).pt
            spec  -> answer states .hf27/ask5/acts_<tag>.pt["Y_new"], ridge init .hf27/ask5/E_<tag>_s5(_shuf).pt
  K         top-K kept "word" tokens per wording before the softmax (st1 used K = 50), T = 1.
Target p (st1_fit.build_p): per wording kept by the junk rule, logits = unembed(unit(Y_k)), keep word tokens
(\\w and not special/whitespace), top K, softmax, average over kept wordings. Prefix split as the fit scripts:
val 0..999, test 1000..1999, train the rest (valid prefixes only).
Fit (st1_p3v): per layer, h -> (h - mx) W + b; KL(p || softmax over the FULL vocabulary of tau * unembed(unit(.)));
Adam (lr 3e-3 x std of the init), batch 256, eval every 50 steps, patience 6, max 3000 steps, early stop on val KL;
init from the ridge map.
Variants (--var): ce (main), shuf (targets p permuted across valid prefixes, seed 0; init from the stored
shuffled-answer ridge map), const (W fixed at 0, only the bias b and tau trained; init b = ridge map's my).
Output: .hf27/w2/E_<tag>_<question>_<var>_k<K>.pt (packed like the ridge maps) + .json report.
Usage: python w2_ce_fit.py <model> <topic|spec> <K> [<K> ...] [--var ce|shuf|const] [--smoke]
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
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("question", choices=["topic", "spec"])
ap.add_argument("K", type=int, nargs="+")
ap.add_argument("--var", default="ce", choices=["ce", "shuf", "const"])
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--smoke", action="store_true", help="two layers, 60 steps, output under .hf27/w2/smoke")
ap.add_argument("--save_p", action="store_true", help="also save the target p (ids/val) for this K")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
M = args.model
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt",
         "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]  # 2B: copy of the pod's /root/e1/acts_2b.pt
QD = {"topic": ("ask3", "t3"), "spec": ("ask5", "s5")}[args.question]
YFILE = f"{R}/.hf27/{QD[0]}/acts_{TAG}.pt"
INIT = f"{R}/.hf27/{QD[0]}/E_{TAG}_{QD[1]}{'_shuf' if args.var == 'shuf' else ''}.pt"
OUT = f"{R}/.hf27/w2" + ("/smoke" if args.smoke else "")
os.makedirs(OUT, exist_ok=True)
NV, NT = 1000, 1000
BS, EVAL_EVERY, PATIENCE, MAX_STEPS, LR_REL = 256, 50, 6, 3000, 3e-3
if args.smoke:
    LAYERS, MAX_STEPS = LAYERS[2:4], 60
dev = "cuda"
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

model = get_lite_model(M)
C.setup(model)
Wu, V, d, gamma = C.Wu, C.V, C.d, C.gamma
unit = C.unit
N = C.N
print(f"[w2ce] {M} {args.question} var={args.var} K={args.K} T={args.T}; Y {YFILE}; X {XFILE}; init {INIT}", flush=True)

# ---------------------------------------------------------------- targets (st1_fit.py, verbatim logic)
with torch.no_grad():
    Y = torch.load(YFILE, map_location="cpu")["Y_new"]  # [3, N, d]
    assert Y.shape[1] == N, (Y.shape, N)
    Yk = [unit(Y[k].float()) for k in range(3)]
    del Y
    raw1 = [C.topk(y, 1, m=None)[:, 0] for y in Yk]
    junk = torch.stack([C.junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    T_raw = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
    valid = cnt > 0
    word = C.topk(T_raw, 10)[:, 0]
    wk = keepk.float() / cnt.clamp_min(1).float()
    del T_raw
    keep = ~C.nonword & ~C.junk_tok
    print(f"[w2ce] wordings dropped by junk rule {int(junk.sum())}/{3 * N}; prefixes dropped {int((~valid).sum())}",
          flush=True)


@torch.no_grad()
def build_p(K, T, bs=250):
    S = 3 * K
    ids_all, val_all = [], []
    for i in range(0, N, bs):
        D = torch.zeros(min(bs, N - i), V, device=dev)
        for k in range(3):
            lg = C.logits(Yk[k][i:i + bs]).masked_fill(~keep, float("-inf"))
            vv, ii = lg.topk(K, -1)
            pk = torch.softmax(vv / T, -1)
            D.scatter_add_(1, ii, pk * wk[k, i:i + bs, None])
        vv, ii = D.topk(S, -1)
        ids_all.append(ii.int().cpu()); val_all.append(vv.cpu())
    return torch.cat(ids_all), torch.cat(val_all)


def load_map(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


INIT_MAP = load_map(INIT)
Xs = torch.load(XFILE, map_location="cpu")["X"]
tr_m = valid.cpu().clone(); tr_m[:NV + NT] = False
va_m = valid.cpu().clone(); va_m[NV:] = False
tr_i, va_i = (torch.nonzero(m)[:, 0].to(dev) for m in (tr_m, va_m))
Wk = Wu.float()  # full vocabulary (P3v)

for K in args.K:
    t_all = time.time()
    if os.path.exists(f"{OUT}/E_{TAG}_{args.question}_{args.var}_k{K}.json") and not args.smoke:
        print(f"[w2ce] K={K} exists, skipping", flush=True)
        continue
    ids, val = build_p(K, args.T)
    if args.save_p:
        torch.save(dict(ids=ids, val=val, keep=keep.cpu(), word=word.cpu(), valid=valid.cpu()),
                   f"{OUT}/p_{TAG}_{args.question}_k{K}.pt")
    ent = -(val.clamp_min(1e-30).log() * val).sum(-1)
    print(f"[w2ce] p K={K}: mean entropy {ent[valid.cpu()].mean():.3f} nats, mean support "
          f"{(val > 0).sum(-1).float().mean():.1f}, mean max p {val[:, 0].mean():.3f}; built in {time.time() - t_all:.0f}s",
          flush=True)
    if args.var == "shuf":  # permute targets across valid prefixes
        vi = torch.nonzero(valid.cpu())[:, 0]
        perm = vi[torch.randperm(len(vi), generator=torch.Generator().manual_seed(0))]
        ids, val = ids.clone(), val.clone()
        ids[vi], val[vi] = ids[perm], val[perm]
    Ik, Pv = ids.long().to(dev), val.to(dev)
    Hp = (torch.where(Pv > 0, Pv * Pv.clamp_min(1e-30).log(), torch.zeros_like(Pv))).sum(-1)

    def kl(X, rows, mx, W, b, logt):
        if len(rows) > BS and not torch.is_grad_enabled():
            return sum(kl(X, rows[i:i + BS], mx, W, b, logt) * len(rows[i:i + BS])
                       for i in range(0, len(rows), BS)) / len(rows)
        h = (X[rows] - mx) @ W + b
        lg = logt.exp() * (unit(h) * gamma) @ Wk.T
        lq = lg.gather(1, Ik[rows]) - torch.logsumexp(lg, -1, keepdim=True)
        return (Hp[rows] - (Pv[rows] * lq).sum(-1)).mean()

    maps, report = {}, {}
    for l in LAYERS:
        t0 = time.time()
        X = Xs[l].float().to(dev)
        mx, W0, my = INIT_MAP[l]
        const = args.var == "const"
        W = torch.zeros_like(W0) if const else W0.clone().requires_grad_(True)
        b = my.clone().requires_grad_(True)
        logt = torch.zeros((), device=dev, requires_grad=True)
        groups = [{"params": [b], "lr": LR_REL * my.std().item()}, {"params": [logt], "lr": 1e-2}]
        if not const:
            groups.insert(0, {"params": [W], "lr": LR_REL * W0.std().item()})
        opt = torch.optim.Adam(groups)
        with torch.no_grad():
            sub = tr_i[:2000]
            g = [(kl(X, sub, mx, W, b, torch.tensor(float(a), device=dev)).item(), a) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
            logt.fill_(min(g)[1])
            best = (kl(X, va_i, mx, W, b, logt).item(), 0, W.detach().clone(), b.detach().clone(), logt.item())
        v0 = best[0]
        gen = torch.Generator(device=dev).manual_seed(l)
        step, bad = 0, 0
        while step < MAX_STEPS and bad < PATIENCE:
            perm = tr_i[torch.randperm(len(tr_i), device=dev, generator=gen)]
            for i in range(0, len(perm) - BS + 1, BS):
                loss = kl(X, perm[i:i + BS], mx, W, b, logt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step % EVAL_EVERY == 0:
                    with torch.no_grad():
                        vk = kl(X, va_i, mx, W, b, logt).item()
                    if vk < best[0] - 1e-4:
                        best, bad = (vk, step, W.detach().clone(), b.detach().clone(), logt.item()), 0
                    else:
                        bad += 1
                    if bad >= PATIENCE or step >= MAX_STEPS:
                        break
        vk, bstep, Wb, bb, lt = best
        with torch.no_grad():
            wv = word[va_i]
            a10 = C.agree(C.topk((X[va_i] - mx) @ Wb + bb), wv).float().mean().item()
            a10_0 = C.agree(C.topk((X[va_i] - mx) @ W0 + my), wv).float().mean().item()
        sc = Wb.abs().max().item() or 1.0
        maps[l] = dict(mx=mx.cpu(), W=(Wb / sc).half().cpu(), scale=sc, my=bb.cpu(), c=None, tau=float(lt), step=bstep)
        report[l] = dict(val_kl0=v0, val_kl=vk, step=bstep, logtau=lt, val_a10_init=a10_0, val_a10=a10)
        print(f"[w2ce {args.question} {args.var} K{K}] L{l}: val KL {v0:.3f} -> {vk:.3f} at step {bstep} "
              f"(stopped {step}); log tau {lt:.2f}; val a10 {a10_0:.3f} -> {a10:.3f}; {time.time() - t0:.0f}s", flush=True)
        del W, b, opt, X
        torch.cuda.empty_cache()
    stem = f"{OUT}/E_{TAG}_{args.question}_{args.var}_k{K}"
    torch.save(maps, f"{stem}.pt")
    json.dump({str(k): v for k, v in report.items()}, open(f"{stem}.json", "w"), indent=1)
    print(f"[w2ce] saved {stem}.pt ({time.time() - t_all:.0f}s)", flush=True)
