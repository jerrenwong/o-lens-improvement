"""W7: A-lens-CE with (1) the scoring-filter junk rule for target tokens and (2) optional merging of surface variants.
Copy of scripts/w2/w2_ce_fit.py (var=ce) with only build_p changed; the fit loop, init, split, early stopping and the
whole-vocabulary softmax are verbatim.
  allowed token (target support) = decoded text has a letter or digit (str.isalnum on some char), not special/added
      (auto/filters.special_ids), and contains none of "@", "&", "<", ">" (auto/filters.MARKUP_CHARS + "@").
      Benchmark items are NOT used (no spared set).
  nomerge: per kept wording, softmax over allowed tokens (T=1), top K tokens, renormalise, average over wordings
           (identical to w2's top-K-then-softmax, only the allowed set differs).
  merge:   per kept wording, softmax over allowed tokens (T=1), sum the probability per normal form
           nf(s) = strip whitespace, drop one leading "_"/U+2581/U+0120, lower-case (NO plural stripping);
           give the group's mass to ONE canonical token (member whose text == nf, else member whose text == " "+nf,
           else the group's highest-probability member for that passage/wording); top K groups, renormalise,
           average over wordings.
Wordings dropped by the old junk rule on the raw top-1 (st1) are dropped as before.
Output: .hf27/w7/E_<tag>_<question>_<mode>_k<K>.pt + .json.   --diag: print target diagnostics (val passages) first.
Usage: python w7_fit.py <model> <topic|spec> --mode all|nomerge|mergeall|merge [--K 10] [--T 1] [--diag] [--nofit] [--smoke]
  modes = Experiment 3 columns: all = C2 (no junk filter, no merge), nomerge = C3, mergeall = C4 (merge, no junk
  filter; empty normal forms stay singletons), merge = C5. T = temperature on the answer logits for the target.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.path.insert(0, f"{R}/scripts/perf1")
_fs = importlib.util.spec_from_file_location("w7_filters", f"{R}/auto/filters.py")
FM = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(FM)
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("question", choices=["topic", "spec"])
ap.add_argument("--mode", default="merge", choices=["all", "nomerge", "mergeall", "merge"])
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--diag", action="store_true")
ap.add_argument("--nofit", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
M = args.model
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt",
         "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
QD = {"topic": ("ask3", "t3"), "spec": ("ask5", "s5")}[args.question]
YFILE = f"{R}/.hf27/{QD[0]}/acts_{TAG}.pt"
INIT = f"{R}/.hf27/{QD[0]}/E_{TAG}_{QD[1]}.pt"
OUT = f"{R}/.hf27/w7" + ("/smoke" if args.smoke else "")
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
K = args.K
print(f"[w7] {M} {args.question} mode={args.mode} K={K} T={args.T}; Y {YFILE}; X {XFILE}; init {INIT}", flush=True)

# ---------------------------------------------------------------- token rules
strs = C.strs
SPEC = FM.special_ids(model.tokenizer)
BAD_CH = ("@",) + tuple(FM.MARKUP_CHARS)
allowed_l = [any(c.isalnum() for c in s) and i not in SPEC and not any(c in s for c in BAD_CH) for i, s in enumerate(strs)]
allowed = torch.tensor(allowed_l, device=dev)


def nf7(s):
    s = s.strip()
    if s[:1] in FM.MARKERS:
        s = s[1:]
    return s.lower()


NF = [nf7(s) for s in strs]


def build_groups(allow_l):
    """groups over allowed tokens by nf7 (empty nf -> own group); canonical member: text == nf, else ' '+nf,
    else -1 (= highest-probability member per row). Returns GID [V] (-1 disallowed), NG, CANON [NG], KIND [NG]."""
    key2g, G_l = {}, []
    for i in range(V):
        if allow_l[i]:
            k = NF[i] if NF[i] != "" else f"\x00{i}"
            G_l.append(key2g.setdefault(k, len(key2g)))
        else:
            G_l.append(-1)
    members = [[] for _ in range(len(key2g))]
    for i in range(V):
        if G_l[i] >= 0:
            members[G_l[i]].append(i)
    canon_l, kind_l = [], []
    for g, key in enumerate(key2g):
        if key.startswith("\x00"):
            canon_l.append(members[g][0]); kind_l.append(3); continue
        ex = [i for i in members[g] if strs[i] == key]
        sp = [i for i in members[g] if strs[i] == " " + key]
        if ex:
            canon_l.append(ex[0]); kind_l.append(0)
        elif sp:
            canon_l.append(sp[0]); kind_l.append(1)
        else:
            canon_l.append(-1); kind_l.append(2)
    kc = Counter(kind_l)
    print(f"[w7] groups over {sum(allow_l)} tokens: {len(key2g)} normal forms: (a) exact lower-case no-space token "
          f"{kc[0]}, (b) only ' '+nf token {kc[1]}, (c) neither {kc[2]}, empty-nf singletons {kc[3]}; groups with >1 "
          f"member {sum(len(m) > 1 for m in members)}", flush=True)
    return (torch.tensor(G_l, device=dev), len(key2g), torch.tensor(canon_l, device=dev),
            torch.tensor(kind_l, device=dev))


GROUPS = {"merge": build_groups(allowed_l)}
if args.mode == "mergeall" or args.diag:
    GROUPS["mergeall"] = build_groups([True] * V)
GID, NG, CANON, KIND_G = GROUPS["merge"]
print(f"[w7] allowed tokens (new junk rule) {int(allowed.sum())}/{V}", flush=True)
# token-level attributes for diagnostics (all tokens, nf7 of any token)
ENT = {"quot", "amp", "lt", "gt", "nbsp", "eacute", "apos", "mdash", "ndash", "hellip", "rsquo", "lsquo", "rdquo",
       "ldquo", "copy", "reg", "trade", "middot", "bull", "laquo", "raquo"}
single = torch.tensor([len(n) == 1 for n in NF], device=dev)
markup = torch.tensor([n[:1] in ("&", "<") or any(c in s for c in "&<>") or (s.lstrip().startswith("&") and n.lstrip("&").rstrip(";") in ENT)
                       for s, n in zip(strs, NF)], device=dev)
entname = torch.tensor([n in ENT for n in NF], device=dev)  # bare entity names ("quot") that are not markup themselves
atk = torch.tensor(["@" in s for s in strs], device=dev)
KIND_T = torch.full((V,), -1, device=dev)
KIND_T[GID >= 0] = KIND_G[GID[GID >= 0]]

# ---------------------------------------------------------------- wordings (st1 rule, verbatim)
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
    keep_old = ~C.nonword & ~C.junk_tok
    print(f"[w7] wordings dropped by junk rule {int(junk.sum())}/{3 * N}; prefixes dropped {int((~valid).sum())}",
          flush=True)


@torch.no_grad()
def wording_target(lg, mode, T):
    """lg [B, V] raw logits -> (ids [B, K], p [B, K]) for one wording; also merge stats."""
    if mode == "old":
        vv, ii = lg.masked_fill(~keep_old, float("-inf")).topk(K, -1)
        return ii, torch.softmax(vv / T, -1), None
    al = allowed if mode in ("nomerge", "merge") else None
    p = torch.softmax((lg if al is None else lg.masked_fill(~al, float("-inf"))) / T, -1)
    if mode in ("all", "nomerge"):
        vv, ii = p.topk(K, -1)
        return ii, vv / vv.sum(-1, keepdim=True), None
    GIDm, NG, CANON, KIND_G = GROUPS[mode]
    B = p.shape[0]
    gidx = GIDm.clamp_min(0)[None].expand(B, V)
    gp = torch.zeros(B, NG, device=dev).scatter_add_(1, gidx, p)
    gmax = torch.zeros(B, NG, device=dev).scatter_reduce_(1, gidx, p, reduce="amax", include_self=True)
    vv, gg = gp.topk(K, -1)
    cid = CANON[gg]
    need = cid < 0
    if need.any():  # highest-probability member for groups without a canonical spelling
        ismax = (p == gmax.gather(1, gidx)) & (GIDm >= 0)[None]
        tok_idx = torch.where(ismax, torch.arange(V, device=dev)[None].expand(B, V), torch.full_like(gidx, -1))
        best = torch.full((B, NG), -1, device=dev, dtype=torch.long).scatter_reduce_(1, gidx, tok_idx, reduce="amax")
        cid = torch.where(need, best.gather(1, gg), cid)
    mx = gmax.gather(1, gg)
    st = dict(merged_w=((vv - mx).sum(-1) / vv.sum(-1)), merged_u=(1 - mx / vv.clamp_min(1e-30)).mean(-1), kind=KIND_G[gg],
              vv=vv / vv.sum(-1, keepdim=True))
    return cid, vv / vv.sum(-1, keepdim=True), st


@torch.no_grad()
def build_p(mode, T, rows=None, bs=250, stats=False):
    rows = torch.arange(N) if rows is None else rows
    S = 3 * K
    ids_all, val_all, ST = [], [], dict(merged_w=[], merged_u=[], kmass=[])
    for s0 in range(0, len(rows), bs):
        r = rows[s0:s0 + bs]
        D = torch.zeros(len(r), V, device=dev)
        for k in range(3):
            lg = C.logits(Yk[k][r])
            ii, pk, st = wording_target(lg, mode, T)
            D.scatter_add_(1, ii, pk * wk[k, r.to(dev), None])
            if stats and st is not None:
                kw = keepk[k, r.to(dev)]
                ST["merged_w"].append(st["merged_w"][kw]); ST["merged_u"].append(st["merged_u"][kw])
                ST["kmass"].append(torch.stack([(st["vv"] * (st["kind"] == j)).sum(-1)[kw] for j in range(4)], -1))
        vv, ii = D.topk(S, -1)
        ids_all.append(ii.int().cpu()); val_all.append(vv.cpu())
    if stats and ST["merged_w"]:
        ST = {k: torch.cat(v) for k, v in ST.items()}
    return torch.cat(ids_all), torch.cat(val_all), ST


va_rows = torch.nonzero(valid.cpu()[:NV])[:, 0]
if args.diag:
    print(f"\n[w7 diag] {TAG} {args.question}: target mass shares on val passages (n={len(va_rows)})", flush=True)
    print(f"{'rule':9s} | single-letter | markup(&<>) | entity-name | '@'   | mass on kinds a/b/c (by final token)")
    res = {}
    for mode in ("old", "all", "nomerge", "mergeall", "merge"):
        ids, val, ST = build_p(mode, args.T, va_rows, stats=True)
        I = ids.long().to(dev); P = val.to(dev)
        f = lambda m: (P * m[I].float()).sum(-1).mean().item()  # noqa: E731
        kt = KIND_T[I]
        km = [(P * (kt == j)).sum(-1).mean().item() for j in range(3)]
        res[mode] = dict(ids=ids, val=val, single=f(single), markup=f(markup), ent=f(entname), at=f(atk), kinds=km,
                         junk_new=f(~allowed), junk_old=f(~keep_old))
        print(f"{mode:9s} | {res[mode]['single']:.4f}        | {res[mode]['markup']:.4f}      | {res[mode]['ent']:.4f}      "
              f"| {res[mode]['at']:.4f} | {km[0]:.3f}/{km[1]:.3f}/{km[2]:.3f} | junk(new rule) {res[mode]['junk_new']:.4f} "
              f"junk(old rule) {res[mode]['junk_old']:.4f}", flush=True)
        if mode in ("merge", "mergeall"):
            print(f"[w7 diag {mode}] merged mass (per wording, kept top-K groups): mass-weighted share not on the group's top "
                  f"single token {ST['merged_w'].mean():.4f}; unweighted mean over groups of (1 - max member share) "
                  f"{ST['merged_u'].mean():.4f}; per-wording target mass by canonical kind (a exact / b space / c "
                  f"neither / empty-nf singleton) {ST['kmass'].mean(0).tolist()}", flush=True)
    prefixes = json.load(open(f"{R}/runs/e1/prefixes.json"))

    def show(ids, val, j, n=8):
        return ", ".join(f"{strs[int(t)]!r}:{v:.2f}" for t, v in zip(ids[j][:n], val[j][:n]) if v > 0)
    for j in range(0, len(va_rows), len(va_rows) // 10)[:10]:
        pi = int(va_rows[j])
        print(f"\n--- passage {pi}: {prefixes[pi]['text'][-160:]!r}")
        print(f"  old  : {show(res['old']['ids'], res['old']['val'], j)}")
        for m in ("all", "merge"):
            print(f"  {m:5s}: {show(res[m]['ids'], res[m]['val'], j)}")
    json.dump({m: {k: v for k, v in r.items() if k not in ('ids', 'val')} for m, r in res.items()},
              open(f"{R}/runs/w7/diag_{TAG}_{args.question}.json", "w"), indent=1)
    del res
if args.nofit:
    sys.exit(0)


# ---------------------------------------------------------------- fit (w2_ce_fit.py, verbatim)
def load_map(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


INIT_MAP = load_map(INIT)
Xs = torch.load(XFILE, map_location="cpu")["X"]
tr_m = valid.cpu().clone(); tr_m[:NV + NT] = False
va_m = valid.cpu().clone(); va_m[NV:] = False
tr_i, va_i = (torch.nonzero(m)[:, 0].to(dev) for m in (tr_m, va_m))
Wk = Wu.float()  # full vocabulary (P3v)

t_all = time.time()
if os.path.exists(f"{OUT}/E_{TAG}_{args.question}_{args.mode}_k{K}" + ("" if args.T == 1.0 else f"_t{args.T:g}") + ".json") and not args.smoke:
    print("[w7] exists, skipping", flush=True); sys.exit(0)
ids, val, _ = build_p(args.mode, args.T)
ent = -(val.clamp_min(1e-30).log() * val).sum(-1)
print(f"[w7] p K={K}: mean entropy {ent[valid.cpu()].mean():.3f} nats, mean support "
      f"{(val > 0).sum(-1).float().mean():.1f}, mean max p {val[:, 0].mean():.3f}; built in {time.time() - t_all:.0f}s",
      flush=True)
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
    W = W0.clone().requires_grad_(True)
    b = my.clone().requires_grad_(True)
    logt = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [W], "lr": LR_REL * W0.std().item()},
                            {"params": [b], "lr": LR_REL * my.std().item()}, {"params": [logt], "lr": 1e-2}])
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
    print(f"[w7 {args.question} {args.mode} K{K}] L{l}: val KL {v0:.3f} -> {vk:.3f} at step {bstep} "
          f"(stopped {step}); log tau {lt:.2f}; val a10 {a10_0:.3f} -> {a10:.3f}; {time.time() - t0:.0f}s", flush=True)
    del W, b, opt, X
    torch.cuda.empty_cache()
stem = f"{OUT}/E_{TAG}_{args.question}_{args.mode}_k{K}" + ("" if args.T == 1.0 else f"_t{args.T:g}")
torch.save(maps, f"{stem}.pt")
json.dump({str(k): v for k, v in report.items()}, open(f"{stem}.json", "w"), indent=1)
print(f"[w7] saved {stem}.pt ({time.time() - t_all:.0f}s)", flush=True)
