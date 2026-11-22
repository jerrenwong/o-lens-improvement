"""st1 P3v (copy of st1_p3.py; only change: the decoded softmax runs over the FULL vocabulary, so tokens outside KEEP get target 0 and are pushed down). P3: no push-back. Per layer, train the linear map h -> (h - mx) W + b so that the DECODED output
(final norm + LM head, restricted to the KEEP 'word' tokens, times a learnable scalar inverse temperature tau, which
does not change any ranking) matches the cleaned distribution p (KL(p || q)). Adam; early stopping on KL over the
fit script's validation prefixes (0..999, valid only); init from the current ridge map B0 (.hf27/ask3/E_<tag>_t3.pt).
Train/val/test prefixes and labels as scripts/st1/st1_fit.py (p files written there).
Also: held-out (test prefixes 1000..1999) decode check of each map's OUTPUT vs p (KL, Spearman, top-1 kept) for
B0, P1 T=1, P2 at T and P3 at T, plus the label-free diagnostics.
Usage: python st1_p3.py <model> <T> [<T> ...]
"""
import json
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.argv, ARGS = sys.argv[:2], sys.argv[2:]
TS = [float(t) for t in ARGS] or [1.0]
torch.backends.cuda.matmul.allow_tf32 = True

M = sys.argv[1]
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt", "2b": "/root/e1/acts_2b.pt"}[TAG]
OUT = f"{R}/.hf27/st1"
NV, NT = 1000, 1000
BS, EVAL_EVERY, PATIENCE, MAX_STEPS, LR_REL = 256, 50, 6, 3000, 3e-3
dev = "cuda"
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

model = get_lite_model(M)
C.setup(model)
Wu, V, d, gamma = C.Wu, C.V, C.d, C.gamma
Xs = torch.load(XFILE, map_location="cpu")["X"]


def load_map(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


B0 = load_map(f"{R}/.hf27/ask3/E_{TAG}_t3.pt")
report = {}
import os  # noqa: E402


def oom_retry(fn):
    for a in range(30):
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[p3] OOM, waiting 60 s (try {a})", flush=True)
            time.sleep(60)
    raise RuntimeError("persistent OOM")


for T in TS:
    ts = str(T).replace(".", "")
    if os.path.exists(f"{OUT}/p3v_{TAG}_t{ts}.json"):
        print(f"[p3] T={T} done earlier, skipping", flush=True)
        continue
    P = torch.load(f"{OUT}/p_{TAG}_word_t{ts}.pt")
    ids, val, keep, word, valid = P["ids"].long(), P["val"], P["keep"].to(dev), P["word"].to(dev), P["valid"]
    kidx = torch.arange(V, device=dev)  # P3v: softmax over the FULL vocabulary (p = 0 off KEEP)
    pos = torch.full((V,), -1, dtype=torch.long, device=dev); pos[kidx] = torch.arange(len(kidx), device=dev)
    Wk = Wu[kidx].float()
    tr_m = valid.clone(); tr_m[:NV + NT] = False
    va_m = valid.clone(); va_m[NV:] = False
    te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
    tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].to(dev) for m in (tr_m, va_m, te_m))
    Ik, Pv = pos[ids.to(dev)].clamp_min(0), val.to(dev)  # support in kept-index space (val == 0 where padded)
    assert bool((pos[ids.to(dev)][Pv > 0] >= 0).all())
    Hp = (torch.where(Pv > 0, Pv * Pv.clamp_min(1e-30).log(), torch.zeros_like(Pv))).sum(-1)

    def kl(X, rows, mx, W, b, logt):
        if len(rows) > BS and not torch.is_grad_enabled():  # evaluation: chunk to save memory
            return sum(kl(X, rows[i:i + BS], mx, W, b, logt) * len(rows[i:i + BS])
                       for i in range(0, len(rows), BS)) / len(rows)
        h = (X[rows] - mx) @ W + b
        lg = logt.exp() * (C.unit(h) * gamma) @ Wk.T
        lq = lg.gather(1, Ik[rows]) - torch.logsumexp(lg, -1, keepdim=True)
        return (Hp[rows] - (Pv[rows] * lq).sum(-1)).mean()

    maps = {}
    report[T] = {}
    for l in LAYERS:
        t0 = time.time()
        X = Xs[l].float().to(dev)
        mx, W0, my = B0[l]
        W = W0.clone().requires_grad_(True)
        b = my.clone().requires_grad_(True)
        logt = torch.zeros((), device=dev, requires_grad=True)
        lr = LR_REL * W0.std().item()
        opt = torch.optim.Adam([{"params": [W], "lr": lr}, {"params": [b], "lr": LR_REL * my.std().item()},
                                {"params": [logt], "lr": 1e-2}])
        with torch.no_grad():
            # tau init: best of a coarse grid for the B0 map on train (only the softmax scale; rankings unaffected)
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
                def _step():
                    loss = kl(X, perm[i:i + BS], mx, W, b, logt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                oom_retry(_step)
                step += 1
                if step % EVAL_EVERY == 0:
                    with torch.no_grad():
                        vk = oom_retry(lambda: kl(X, va_i, mx, W, b, logt).item())
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
        sc = Wb.abs().max().item()
        maps[l] = dict(mx=mx.cpu(), W=(Wb / sc).half().cpu(), scale=sc, my=bb.cpu(), c=None, tau=float(lt), step=bstep)
        report[T][l] = dict(val_kl0=v0, val_kl=vk, step=bstep, logtau=lt, val_a10_B0=a10_0, val_a10=a10)
        print(f"[p3 T{T}] L{l}: val KL {v0:.3f} -> {vk:.3f} at step {bstep} (stopped {step}); log tau {lt:.2f}; "
              f"val a10 {a10_0:.3f} -> {a10:.3f}; {time.time() - t0:.0f}s", flush=True)
        del W, b, opt, X
        torch.cuda.empty_cache()
    torch.save(maps, f"{OUT}/E_{TAG}_p3v_t{ts}.pt")
    del Wk
    torch.cuda.empty_cache()
    # held-out decode check of map outputs vs p (same T) + diagnostics, for B0 / P1 T=1 / P2 T / P3 T
    names = {"B0": f"{R}/.hf27/ask3/E_{TAG}_t3.pt", "p1_t10": f"{OUT}/E_{TAG}_p1_t10.pt",
             f"p2_t{ts}": f"{OUT}/E_{TAG}_p2_t{ts}.pt", f"p3v_t{ts}": f"{OUT}/E_{TAG}_p3v_t{ts}.pt"}
    print(f"\n== held-out map OUTPUT vs p(T={T}) on test prefixes (KL over KEEP at natural logit scale; Spearman; "
          f"top-1 kept) + diagnostics (nw1 nw10 fn1 | a1u a1 a10)")
    report[T]["heldout"] = {}
    for nm, pth in names.items():
        mp = load_map(pth)
        rows = []
        for l in LAYERS:
            X = Xs[l][te_i.cpu()].float().to(dev)
            mx, W, my = mp[l]
            pred = (X - mx) @ W + my
            full = torch.zeros(C.N, d); full[te_i.cpu()] = pred.cpu()
            dc = oom_retry(lambda: C.decode_check(f"{nm} L{l}", full, ids, val, keep, te_i.cpu(), quiet=True, sp_n=300))
            dg = oom_retry(lambda: C.diag(pred, word[te_i]))
            rows.append({**dc, **dg})
        m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        report[T]["heldout"][nm] = dict(mean=m, per_layer=rows)
        print(f"   {nm:8s} mean over layers: KL {m['kl']:.3f} Spearman {m['spearman']:.3f} top-1 kept {m['top1']:.3f} | "
              f"nw1 {m['nw1']:.3f} nw10 {m['nw10']:.3f} fn1 {m['fn1']:.3f} | a1u {m['a1u']:.3f} a1 {m['a1']:.3f} a10 {m['a10']:.3f}",
              flush=True)
    json.dump({str(l): v for l, v in report[T].items()}, open(f"{OUT}/p3v_{TAG}_t{ts}.json", "w"), indent=1)
json.dump({str(k): {str(l): v for l, v in x.items()} for k, x in report.items()},
          open(f"{OUT}/p3v_{TAG}_{'_'.join(ARGS) or '1.0'}.json", "w"), indent=1)
print("saved", flush=True)
