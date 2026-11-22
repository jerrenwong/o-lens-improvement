"""rb1 P3v: scripts/st1/st1_p3v.py (T=1) on a SUBSET of the training prefixes. Only changes: (1) the training rows are
the rb1 subset (.hf27/rb1/subsets_<tag>.pt) instead of all train prefixes; (2) initialised from the ridge map refitted
on that same subset (.hf27/rb1/E_<tag>_<ridge var>.pt) instead of the stored full-data ridge map (st1 inits from its
own ridge map; this keeps the subset map free of the other prefixes). Loss, full-vocabulary softmax, tau, Adam, lr,
batch, eval cadence, patience, early stopping on validation KL: unchanged.
Var "pall" = the stored st1 map (.hf27/st1/E_<tag>_p3v_t10.pt), no training: only the held-out diagnostics.
Writes .hf27/rb1/E_<tag>_p<n>_<k>.pt and .hf27/rb1/p3v_<tag>.json (val KL, step, held-out decode check + diag).
Usage: python rb1_p3v.py <model> <ridge var> [...]   e.g. r1000_0 r1000_1 r10000_0 r10000_1 pall
"""
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.path.insert(0, f"{R}/scripts/perf1")
torch.backends.cuda.matmul.allow_tf32 = True
M, VARS = sys.argv[1], sys.argv[2:]
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt", "2b": "/root/e1/acts_2b.pt"}[TAG]
OUT = f"{R}/.hf27/rb1"
NV, NT = 1000, 1000
BS, EVAL_EVERY, PATIENCE, MAX_STEPS, LR_REL = 256, 50, 6, 3000, 3e-3
dev = "cuda"
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

model = get_lite_model(M)
C.setup(model)
Wu, V, d, gamma = C.Wu, C.V, C.d, C.gamma
Xs = torch.load(XFILE, map_location="cpu")["X"]
SUB = torch.load(f"{OUT}/subsets_{TAG}.pt")


def load_map(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


P = torch.load(f"{R}/.hf27/st1/p_{TAG}_word_t10.pt")
ids, val, keep, word, valid = P["ids"].long(), P["val"], P["keep"].to(dev), P["word"].to(dev), P["valid"]
kidx = torch.arange(V, device=dev)
pos = torch.full((V,), -1, dtype=torch.long, device=dev); pos[kidx] = torch.arange(len(kidx), device=dev)
Wk = Wu[kidx].float()
va_m = valid.clone(); va_m[NV:] = False
te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
va_i, te_i = (torch.nonzero(m)[:, 0].to(dev) for m in (va_m, te_m))
assert torch.equal(va_i.cpu(), SUB["va_i"]) and torch.equal(te_i.cpu(), SUB["te_i"])
Ik, Pv = pos[ids.to(dev)].clamp_min(0), val.to(dev)
Hp = (torch.where(Pv > 0, Pv * Pv.clamp_min(1e-30).log(), torch.zeros_like(Pv))).sum(-1)


def kl(X, rows, mx, W, b, logt):
    if len(rows) > BS and not torch.is_grad_enabled():
        return sum(kl(X, rows[i:i + BS], mx, W, b, logt) * len(rows[i:i + BS])
                   for i in range(0, len(rows), BS)) / len(rows)
    h = (X[rows] - mx) @ W + b
    lg = logt.exp() * (C.unit(h) * gamma) @ Wk.T
    lq = lg.gather(1, Ik[rows]) - torch.logsumexp(lg, -1, keepdim=True)
    return (Hp[rows] - (Pv[rows] * lq).sum(-1)).mean()


jf = f"{OUT}/p3v_{TAG}.json"
report = json.load(open(jf)) if os.path.exists(jf) else {}
for var in VARS:
    name = "pall" if var == "pall" else "p" + var[1:]
    if name in report:
        print(f"[rb1 p3v] {name} done earlier, skipping", flush=True)
        continue
    rep = {"train": {}}
    if var == "pall":
        mp_path = f"{R}/.hf27/st1/E_{TAG}_p3v_t10.pt"
    else:
        tr_i = SUB["subsets"][var].to(dev)
        B0 = load_map(f"{OUT}/E_{TAG}_{var}.pt")
        maps = {}
        for l in LAYERS:
            t0 = time.time()
            X = Xs[l].float().to(dev)
            mx, W0, my = B0[l]
            W = W0.clone().requires_grad_(True)
            b = my.clone().requires_grad_(True)
            logt = torch.zeros((), device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [W], "lr": LR_REL * W0.std().item()},
                                    {"params": [b], "lr": LR_REL * my.std().item()},
                                    {"params": [logt], "lr": 1e-2}])
            with torch.no_grad():
                sub = tr_i[:2000]
                g = [(kl(X, sub, mx, W, b, torch.tensor(float(a), device=dev)).item(), a) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
                logt.fill_(min(g)[1])
                best = (kl(X, va_i, mx, W, b, logt).item(), 0, W.detach().clone(), b.detach().clone(), logt.item())
            v0 = best[0]
            gen = torch.Generator(device=dev).manual_seed(l)
            step, bad = 0, 0
            torch.set_grad_enabled(True)
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
            torch.set_grad_enabled(False)
            vk, bstep, Wb, bb, lt = best
            sc = Wb.abs().max().item()
            maps[l] = dict(mx=mx.cpu(), W=(Wb / sc).half().cpu(), scale=sc, my=bb.cpu(), c=None, tau=float(lt), step=bstep)
            rep["train"][str(l)] = dict(val_kl0=v0, val_kl=vk, step=bstep, stopped=step, logtau=lt)
            print(f"[rb1 p3v {name}] L{l}: val KL {v0:.3f} -> {vk:.3f} at step {bstep} (stopped {step}); {time.time() - t0:.0f}s",
                  flush=True)
            del W, b, opt, X
            torch.cuda.empty_cache()
        mp_path = f"{OUT}/E_{TAG}_{name}.pt"
        torch.save(maps, mp_path)
    torch.set_grad_enabled(False)
    mp = load_map(mp_path)
    rows = []
    for l in LAYERS:
        X = Xs[l].float().to(dev)
        mx, W, my = mp[l]
        vkl = kl(X, va_i, mx, W, my, torch.zeros((), device=dev)).item()  # tau-free val KL (full vocab, tau=1)
        pred = (X[te_i] - mx) @ W + my
        full = torch.zeros(C.N, d); full[te_i.cpu()] = pred.cpu()
        dc = C.decode_check(f"{name} L{l}", full, ids, val, keep, te_i.cpu(), quiet=True, sp_n=300)
        dg = C.diag(pred, word[te_i])
        rows.append(dict(val_kl_tau1=vkl, **dc, **dg))
        del X
    m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    rep["heldout_mean"], rep["heldout"] = m, rows
    rep["n"] = len(SUB["subsets"]["rall" if var == "pall" else var])
    print(f"[rb1 p3v {name}] held-out mean: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
    report[name] = rep
    json.dump(report, open(jf, "w"), indent=1)
print("saved", flush=True)
