"""st1 fit: cleaned-distribution targets for the ask-map (P1 expected unembedding, P2 least-squares push-back),
same ridge, grid, validation rule and train/val/test prefixes as scripts/ask3/ask3_fit9.py (m2b_fit.py for 2B).

Cleaned distribution p (per prefix): for each wording k kept by the fit script's junk rule (unmasked top-1 of
unembed(Y_k) not special/added/whitespace), logits = unembed(Y_k) (= final norm + LM head); keep only KEEP tokens
(run A "word": token string contains a \\w character AND not a special/added/whitespace-only token; run B "wordnf":
A minus the fit script's function-word list); top K=50; softmax(logits/T); average the kept wordings' distributions
(weights 1/#kept). T=0: one-hot on each wording's top-1 kept token, averaged.
P1: z = sum_i p_i u_i/|u_i| (u_i = LM-head rows); y = z / gamma (gamma = final-norm gain incl. sqrt(1+eps)), so
    final_norm(y) = gamma * y / rms(y) is proportional to z and unembed(y) = W z / rms(y). target = unit(y).
P2: t = log p on the support S of p, tau = min_S log p - 1 on every other KEEP token, centred over KEEP;
    z = argmin |W_keep z - t|^2 + rho |z|^2 (one Cholesky of W_keep^T W_keep + rho I); y = z / gamma; unit(y).
Validation labels: the fit script's elicited word (masked top-1 of the junk-dropped raw target) and its valid mask,
identical for every variant, so all maps are selected by the same rule as B0.
Usage: python st1_fit.py <model>
"""
import json
import os
import re
import sys
from collections import Counter

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt", "2b": "/root/e1/acts_2b.pt"}[TAG]
OUT = f"{R}/.hf27/st1"
os.makedirs(OUT, exist_ok=True)
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
K = 50
RHOS = [1e-4, 1e-3, 1e-2, 1e-1]
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


data = json.load(open(f"{R}/runs/e1/prefixes.json"))
N = len(data)
model = get_lite_model(M)
tok = model.tokenizer
Wu = model._lm_head.weight  # [V, d] bf16
V, d = Wu.shape
strs = [tok.decode([i]) for i in range(V)]
nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
mask = nonword | func  # fit script's validation mask
norm_s = [s.strip().lower() for s in strs]
vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
special_ids = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special_ids |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
junk_tok[[i for i in special_ids if i < V]] = True
junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)
KEEP = {"word": ~nonword & ~junk_tok, "wordnf": ~nonword & ~junk_tok & ~func}
print(f"[st1] {M}: V {V} d {d}; nonword {int(nonword.sum())}, junk {int(junk_tok.sum())}, func {int(func.sum())}; "
      f"KEEP word {int(KEEP['word'].sum())}, wordnf {int(KEEP['wordnf'].sum())}; "
      f"special that contain \\w (dropped by junk rule, not by nonword): {int((junk_tok & ~nonword).sum())}", flush=True)


def head_gamma():
    norm = model._final_norm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    ones = torch.ones(1, d, dtype=torch.float32, device=dev)
    return (norm(ones).float()[0] * (1.0 + eps) ** 0.5)


gamma = head_gamma()
print(f"[st1] final-norm gain gamma: min {gamma.min():.4f} max {gamma.max():.4f} mean {gamma.mean():.4f}; "
      f"|gamma|<0.05: {int((gamma.abs() < 0.05).sum())}", flush=True)
# check: unembed(v) == W (gamma * v / rms(v)) up to bf16
v = torch.randn(4, d, device=dev)
a = model.unembed(v).float()
b = (Wu.float() @ (gamma * unit(v)).T).T
print(f"[st1] decode formula check: max |diff| {(a - b).abs().max():.3f} (logit scale {a.abs().max():.1f}); "
      f"top-1 equal {(a.argmax(-1) == b.argmax(-1)).tolist()}", flush=True)


def logits(v, bs=250):
    return torch.cat([model.unembed(v[i:i + bs].to(dev).float()).float() for i in range(0, v.shape[0], bs)])


def topk(v, k=10, bs=250, m=mask):
    out = []
    for i in range(0, v.shape[0], bs):
        lg = model.unembed(v[i:i + bs].to(dev).float()).float()
        if m is not None:
            lg = lg.masked_fill(m, float("-inf"))
        out.append(lg.topk(k, -1).indices)
    return torch.cat(out)


def agree(top, word):
    return (NORM[top] == NORM[word][:, None]).any(-1)


# ------------------------------------------------------------------ raw targets, junk rule, labels (as fit script)
Y = torch.load(f"{R}/.hf27/ask3/acts_{TAG}.pt", map_location="cpu")["Y_new"]  # [3, N, d]
Yk = [unit(Y[k].float()) for k in range(3)]
del Y
raw1 = [topk(y, 1, m=None)[:, 0] for y in Yk]
junk = torch.stack([junk_tok[r] for r in raw1])  # [3, N]
keepk = ~junk
cnt = keepk.sum(0)
T_raw = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
valid = cnt > 0
word = topk(T_raw, 10)[:, 0]
wk = keepk.float() / cnt.clamp_min(1).float()  # [3, N] wording weights
print(f"[st1] wordings dropped by junk rule {int(junk.sum())}/{3 * N}; prefixes dropped {int((~valid).sum())}", flush=True)

# ------------------------------------------------------------------ cleaned distributions p (sparse, support <= 3K)
S = 3 * K


def build_p(keep, T, bs=250):
    ids_all, val_all = [], []
    for i in range(0, N, bs):
        D = torch.zeros(min(bs, N - i), V, device=dev)
        for k in range(3):
            lg = logits(Yk[k][i:i + bs]).masked_fill(~keep, float("-inf"))
            vv, ii = lg.topk(K, -1)
            if T == 0:
                pk = torch.zeros_like(vv); pk[:, 0] = 1
            else:
                pk = torch.softmax(vv / T, -1)
            D.scatter_add_(1, ii, pk * wk[k, i:i + bs, None])
        vv, ii = D.topk(S, -1)
        ids_all.append(ii.int().cpu()); val_all.append(vv.cpu())
    return torch.cat(ids_all), torch.cat(val_all)


U_norm = Wu.float().norm(dim=-1)  # [V]


def p1_target(ids, val, bs=500):
    out = []
    for i in range(0, N, bs):
        I, P = ids[i:i + bs].long().to(dev), val[i:i + bs].to(dev)
        z = torch.einsum("bs,bsd->bd", P / U_norm[I], Wu[I].float())
        out.append(unit(z / gamma).cpu())
    return torch.cat(out)


class P2:
    def __init__(self, keep):
        self.kidx = keep.nonzero()[:, 0]
        Wk = Wu[self.kidx].float()
        G = (Wk.T @ Wk).double()
        self.sumW = Wk.sum(0)
        self.Vw = len(self.kidx)
        del Wk
        self.ev = torch.linalg.eigvalsh(G).mean().item()
        self.G = G
        self.L = {}

    def solve(self, ids, val, rho, rows=None, bs=500):
        if rho not in self.L:
            self.L[rho] = torch.linalg.cholesky(self.G + rho * self.ev * torch.eye(d, device=dev, dtype=torch.float64))
        Lc = self.L[rho]
        rows = torch.arange(N) if rows is None else rows
        out = []
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            I, P = ids[r].long().to(dev), val[r].to(dev)
            sup = P > 0
            lp = torch.where(sup, P.clamp_min(1e-30).log(), torch.zeros_like(P))
            tau = torch.where(sup, lp, torch.full_like(lp, float("inf"))).min(-1).values - 1.0
            a = torch.where(sup, lp - tau[:, None], torch.zeros_like(lp))  # >= 1 on the support
            rhs = torch.einsum("bs,bsd->bd", a, Wu[I].float()) - (a.sum(-1, keepdim=True) / self.Vw) * self.sumW
            z = torch.cholesky_solve(rhs.double().T, Lc).T.float()
            out.append(unit(z / gamma).cpu())
        return torch.cat(out)


def avg_rank(x):
    o = x.argsort(); r = torch.empty_like(x); r[o] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    u, inv = torch.unique(x, return_inverse=True)  # average ranks over ties
    s = torch.zeros(len(u), device=x.device, dtype=x.dtype).scatter_add_(0, inv, r)
    c = torch.zeros(len(u), device=x.device, dtype=x.dtype).scatter_add_(0, inv, torch.ones_like(r))
    return (s / c)[inv]


def decode_check(name, tgt, ids, val, keep, rows):
    """Decode the TARGET (or map output) and compare with p over the KEEP tokens:
    KL(p || q), q = softmax of decoded logits renormalised over KEEP (natural logit scale, no temperature);
    Spearman between p and decoded scores on S u decoded top-50 (p = 0 off-support, ties averaged);
    top-1 of p stays top-1 after decoding (KEEP-masked); overlap of decoded top-10 with p's top-10."""
    lg = logits(tgt[rows]).masked_fill(~keep, float("-inf"))
    lq = torch.log_softmax(lg, -1)
    I, P = ids[rows].long().to(dev), val[rows].to(dev)
    sup = P > 0
    kl = (torch.where(sup, P * (P.clamp_min(1e-30).log() - lq.gather(1, I)), torch.zeros_like(P))).sum(-1).mean().item()
    t1 = (lg.argmax(-1) == I[:, 0]).float().mean().item()
    d10 = lg.topk(10, -1).indices
    d50 = lg.topk(50, -1).indices
    nsup = sup.sum(-1).clamp(max=10)
    ov = torch.stack([(torch.isin(d10[j], I[j, :nsup[j]])).sum() / nsup[j] for j in range(len(rows))]).float().mean().item()
    sc = []
    for j in range(len(rows)):
        u = torch.unique(torch.cat([I[j][sup[j]], d50[j]]))
        pv = torch.zeros(V, device=dev); pv[I[j][sup[j]]] = P[j][sup[j]]
        a, b = avg_rank(lg[j, u]), avg_rank(pv[u])
        if b.std() > 0:
            sc.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    sp = sum(sc) / max(1, len(sc))
    print(f"   decode check {name:14s}: KL(p||q) {kl:.3f}; Spearman {sp:.3f}; top-1 kept {t1:.3f}; top-10 overlap {ov:.3f}"
          f" (n={len(rows)})", flush=True)
    return dict(kl=kl, spearman=sp, top1=t1, overlap10=ov)


chk_rows = torch.arange(NV, NV + NT)  # held-out test prefixes of the fit script
targets, info, pstore = {}, {}, {}
PLAN = {"word": [0.5, 1.0, 2.0, 0.0], "wordnf": [1.0]}  # p built for these; P1 target only at T=1 (+ T=0 ref)
for run, Ts in PLAN.items():
    keep = KEEP[run]
    for T in Ts:
        ids, val = build_p(keep, T)
        pstore[(run, T)] = (ids, val)
        ent = -(val.clamp_min(1e-30).log() * val).sum(-1)
        sup = (val > 0).sum(-1).float()
        print(f"[st1] p {run} T={T}: mean entropy {ent.mean():.3f} nats, mean support {sup.mean():.1f}, "
              f"mean max p {val[:, 0].mean():.3f}; argmax p == fit word {(NORM[ids[:, 0].long().to(dev)] == NORM[word]).float().mean():.3f}", flush=True)
        torch.save(dict(ids=ids, val=val, keep=keep.cpu(), word=word.cpu(), valid=valid.cpu()),
                   f"{OUT}/p_{TAG}_{run}_t{str(T).replace('.', '')}.pt")
        if T in (1.0, 0.0):
            nm = f"p1{'' if run == 'word' else 'f'}_t{str(T).replace('.', '')}"
            targets[nm] = p1_target(ids, val)
            info[nm] = decode_check(nm, targets[nm], ids, val, keep, chk_rows)
# P2 (run A, T in {0.5, 1, 2}); rho by decode fidelity on 1000 fit-train prefixes (ties -> larger rho)
p2 = P2(KEEP["word"])
rows_rho = torch.arange(NV + NT, NV + NT + 1000)
ids, val = pstore[("word", 1.0)]
rsc = {}
for rho in RHOS:
    t = p2.solve(ids, val, rho, rows_rho)
    tt = torch.zeros(N, d); tt[rows_rho] = t
    c = decode_check(f"p2 rho{rho:g}", tt, ids, val, KEEP["word"], rows_rho)
    rsc[rho] = c["top1"] + c["overlap10"]
RHO = max(RHOS, key=lambda r: (round(rsc[r], 4), r))
print(f"[st1] P2 rho = {RHO} x mean eigenvalue ({p2.ev:.3g})", flush=True)
for run, T in (("word", 0.5), ("word", 1.0), ("word", 2.0), ("wordnf", 1.0)):
    ids, val = pstore[(run, T)]
    nm = f"p2{'' if run == 'word' else 'f'}_t{str(T).replace('.', '')}"
    if run == "wordnf" and not hasattr(p2, "nf"):
        p2 = P2(KEEP["wordnf"]); p2.nf = True
    targets[nm] = p2.solve(ids, val, RHO)
    info[nm] = decode_check(nm, targets[nm], ids, val, KEEP[run], chk_rows)
del p2
torch.cuda.empty_cache()
# raw decode check for reference: how well does the RAW target rank p (T=1)
ids, val = pstore[("word", 1.0)]
for T in (0.5, 1.0, 2.0):
    ids, val = pstore[("word", T)]
    info[f"raw_t{str(T).replace('.', '')}"] = decode_check(f"raw vs p T{T}", T_raw, ids, val, KEEP["word"], chk_rows)
targets["raw"] = T_raw  # refit check: should reproduce B0's c and scores


# ------------------------------------------------------------------ ridge (identical to fit script)
class Ridge:
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

    def predict(self, X, W):
        return (X - self.mx) @ W + self.my


def pack(r, W, c):
    sc = W.abs().max().item()
    return dict(mx=r.mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=r.my.cpu(), c=c)


tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].cpu() for m in (tr_m, va_m, te_m))
print(f"[st1] train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}", flush=True)
B0 = torch.load(f"{R}/.hf27/ask3/E_{TAG}_t3.pt", map_location="cpu")
B0 = {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float(), v["c"]) for l, v in B0.items()}
Xs = torch.load(XFILE, map_location="cpu")["X"]


def diag(pred, wt):
    """label-free diagnostics on held-out prefixes: non-word share (nonword or special/whitespace) of unmasked
    top-1 / top-10; function-word share of unmasked top-1; agreement of top-1 (unmasked / masked) and top-10
    (masked) with the model's elicited answer word."""
    tu = topk(pred, 10, m=None)
    tm = topk(pred, 10)
    nw = nonword | junk_tok
    return dict(nw1=nw[tu[:, 0]].float().mean().item(), nw10=nw[tu].float().mean().item(),
                fn1=func[tu[:, 0]].float().mean().item(),
                a1u=(NORM[tu[:, 0]] == NORM[wt]).float().mean().item(),
                a1=agree(tm[:, :1], wt).float().mean().item(), a10=agree(tm, wt).float().mean().item())


maps = {nm: {} for nm in targets}
res = {nm: {} for nm in ["B0"] + list(targets)}
for l in LAYERS:
    X = Xs[l].float().to(dev)
    r = Ridge(X[tr_i.to(dev)])
    wv, wt = word[va_i.to(dev)], word[te_i.to(dev)]
    mx, W0, my, c0 = B0[l]
    res["B0"][l] = dict(c=c0, **diag((X[te_i.to(dev)] - mx) @ W0 + my, wt))
    for nm, Tg in targets.items():
        r.setY(Tg[tr_i].to(dev))
        sc = [agree(topk(r.predict(X[va_i.to(dev)], r.W(c_))), wv).float().mean().item() for c_ in GRID]
        b = max(range(len(GRID)), key=lambda i: (sc[i], i))
        W = r.W(GRID[b])
        if nm != "raw":
            maps[nm][l] = pack(r, W, GRID[b])
        res[nm][l] = dict(c=GRID[b], val10=sc[b], **diag(r.predict(X[te_i.to(dev)], W), wt))
        del W
    del r, X
    torch.cuda.empty_cache()
    print(f"L{l}: " + "  ".join(f"{nm} c={res[nm][l]['c']:g} a10={res[nm][l]['a10']:.3f} nw1={res[nm][l]['nw1']:.3f}"
                             for nm in res), flush=True)

print("\n== held-out (fit-script test prefixes) diagnostics per layer: nw1 nw10 fn1 | a1u a1 a10 (answer word)")
for nm in res:
    print(f"-- {nm}")
    for l in LAYERS:
        x = res[nm][l]
        print(f"   L{l:2d} c={x['c']:<6g} nw1 {x['nw1']:.3f} nw10 {x['nw10']:.3f} fn1 {x['fn1']:.3f} | "
              f"a1u {x['a1u']:.3f} a1 {x['a1']:.3f} a10 {x['a10']:.3f}")
    m = {k: sum(res[nm][l][k] for l in LAYERS) / len(LAYERS) for k in ("nw1", "nw10", "fn1", "a1u", "a1", "a10")}
    print(f"   mean   nw1 {m['nw1']:.3f} nw10 {m['nw10']:.3f} fn1 {m['fn1']:.3f} | a1u {m['a1u']:.3f} a1 {m['a1']:.3f} a10 {m['a10']:.3f}")
for nm, m in maps.items():
    torch.save(m, f"{OUT}/E_{TAG}_{nm}.pt")
json.dump(dict(decode=info, diag={k: {str(l): v for l, v in x.items()} for k, x in res.items()}, rho=RHO),
          open(f"{OUT}/fit_{TAG}.json", "w"), indent=1)
print("saved", flush=True)
