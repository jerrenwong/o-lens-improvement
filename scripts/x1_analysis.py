"""x1: context-routed readout of J++ experts - sanity checks, router assignment of benchmark rows (FIT + DEV only),
distance to centroids vs WikiText, qualitative top-8 (DEV prompts). Never touches test.
    python scripts/x1_analysis.py > /root/x1_analysis.log"""
import json
import math
import sys
from collections import Counter, defaultdict

import torch as t

sys.path.insert(0, "/workspace/o-lens-improvement/src")
sys.path.insert(0, "/workspace/o-lens-improvement/auto/candidates")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import get_position_mask_with_early_skips, load_lens_file, record_activations  # noqa
from olens.readout import Bench, candidate_ranks, unembed_logits  # noqa: E402

R = "/workspace/o-lens-improvement"
R2 = f"{R}/runs/r2_9b"
M = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
t.set_grad_enabled(False)
dev = "cuda"


def P(*a):
    print(*a, flush=True)


def lensd(p):
    return {int(l): v.float() for l, v in load_lens_file(p, hf_model_name=M).jacobians_L_dict_FN.items()}


# ------------------------------------------------------------- 1. sanity (CPU)
P("== SANITY")
for c in ("A", "C"):
    ex = t.load(f"{R2}/{c}/{c}_full/experts_K8.pt", map_location="cpu", weights_only=True)
    W = json.load(open(f"{R2}/{c}/{c}_full/jpp_{c}_full_expert_weights.json"))["by_layer"]
    fixed = lensd(f"{R2}/{c}/{c}_full/jpp_{c}_full.pt")
    pooled = lensd(f"{R2}/{c}/{c}_full/pooled_{c}_full.pt")
    for l in LAYERS:
        E = ex["experts"][l].to(dev).float()
        w = t.tensor(W[str(l)]["weights_unit_l2"], device=dev)
        rebuilt = t.einsum("e,eij->ij", w, E)
        F = fixed[l].to(dev)
        cnt = ex["position_counts"][l].to(dev).float()
        um = E.mean(0)
        cm = (cnt[:, None, None] * E).sum(0) / cnt.sum()
        Pl = pooled[l].to(dev)
        sc = F.abs().max().item(); scp = Pl.abs().max().item()
        cos = lambda a, b: float((a * b).sum() / a.norm() / b.norm())  # noqa: E731
        P(f"{c} L{l:2d} fixed: max|rebuilt-file| {(rebuilt - F).abs().max().item():.2e} (max|J| {sc:.2e}) cos {cos(rebuilt, F):.6f}"
          f" | pooled: unweighted-mean max diff {(um - Pl).abs().max().item():.2e} cos {cos(um, Pl):.6f};"
          f" count-weighted {(cm - Pl).abs().max().item():.2e} cos {cos(cm, Pl):.6f}; file 'pooled' key"
          f" {(ex['pooled'][l].to(dev).float() - Pl).abs().max().item():.2e} (max|P| {scp:.2e})"
          f" | counts {cnt.long().tolist()} | w {[round(x, 2) for x in w.tolist()]}")
        del E
    del ex
    t.cuda.empty_cache()

# ------------------------------------------------------------- 2. router assignment
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(b.num_items) if (b.eval_slugs[i], b.item_names[i]) in part]
spl = [part[(b.eval_slugs[i], b.item_names[i])] for i in idx]
task = [b.eval_slugs[i] for i in idx]
if b.readout_positions is not None:
    for s in ("fit", "dev"):
        pos = [b.readout_positions[i] for i, q in zip(idx, spl) if q == s]
        pos.sort()
        P(f"readout positions {s}: n {len(pos)} median {pos[len(pos)//2]} p10 {pos[len(pos)//10]} p90 {pos[9*len(pos)//10]}")

routers = {}
for c in ("A", "C"):
    rs = t.load(f"{R2}/{c}/router_K8.pt", map_location="cpu", weights_only=True)
    routers[c] = {int(l): (rs["pca_means"][l].to(dev).float(), rs["pca_components"][l].to(dev).float(),
                           rs["centroids"][l].to(dev).float()) for l in rs["centroids"]}


def route(c, l, h):
    m, Pc, C = routers[c][l]
    z = (h.float() / h.float().norm(dim=-1, keepdim=True) - m) @ Pc
    d = t.cdist(z, C)
    return d.argmin(-1), d.min(-1).values


model = get_hf_model(M, attn_implementation="sdpa")
pools = {"A": json.load(open(f"{R}/runs/r1_2b/prompts_raw.json")),
         "C": json.load(open(f"{R}/runs/r1_2b/prompts_natural.json"))}
NW = 48
wk = {}  # c -> {l: (assign, dist, pos)}
for c in ("A", "C"):
    acc = defaultdict(list)
    for p in pools[c][:NW]:
        ids, act = record_activations(model, p, 128, LAYERS)
        S = ids.shape[1]
        pos = t.arange(S)
        for l in LAYERS:
            a, d = route(c, l, act[l].to(dev))
            acc[l].append((a.cpu(), d.cpu(), pos))
    wk[c] = {l: tuple(t.cat([x[j] for x in acc[l]]) for j in range(3)) for l in LAYERS}


def nmi(x, y):
    n = len(x); cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    I = sum(v / n * math.log(v * n / (cx[a] * cy[b_])) for (a, b_), v in cxy.items())
    H = lambda cc: -sum(v / n * math.log(v / n) for v in cc.values())  # noqa: E731
    return I / max(1e-12, math.sqrt(H(cx) * H(cy)))


P("\n== ROUTER ASSIGNMENT (cluster counts per task; WikiText = router's own pool, 48 prompts)")
summary = {}
for c in ("A", "C"):
    skip = 16 if c == "A" else 1
    for l in LAYERS:
        h = b.resid[l][idx].to(dev)
        a, d = route(c, l, h)
        a, d = a.cpu().tolist(), d.cpu()
        wa, wd, wp = wk[c][l]
        valid = (wp >= skip) & (wp < wp.new_tensor(127))
        wa_v = wa[valid].tolist()
        wcnt = Counter(wa_v)
        P(f"\n[{c} L{l}] cluster:        " + " ".join(f"{k:>4d}" for k in range(8)) + "   NMI(task,cluster) dev+fit")
        P(f"  WikiText % (pos>={skip}) " + " ".join(f"{100*wcnt[k]/len(wa_v):4.0f}" for k in range(8)))
        for s in ("fit", "dev"):
            for tk in TASKS:
                cc = Counter(a_ for a_, q, t_ in zip(a, spl, task) if q == s and t_ == tk)
                P(f"  {s} {tk[:12]:12s} " + " ".join(f"{cc[k]:4d}" for k in range(8)))
        nm = nmi(task, a)
        P(f"  NMI(task, cluster) = {nm:.2f}")
        # distances
        wd_v = wd[valid]
        q = lambda x, p: float(t.quantile(x.float(), p))  # noqa: E731
        early = wd[(wp >= 1) & (wp < 16)]
        line = (f"  dist to nearest centroid: WikiText(pos>={skip}) med {q(wd_v, .5):.3f} p90 {q(wd_v, .9):.3f} p99 {q(wd_v, .99):.3f}"
                f" | WikiText pos1-15 med {q(early, .5):.3f}")
        for s in ("fit", "dev"):
            ds = t.tensor([float(x) for x, q_ in zip(d, spl) if q_ == s])
            line += f" | {s} med {q(ds, .5):.3f} >p99wk {100*float((ds > q(wd_v, .99)).float().mean()):.0f}%"
        P(line)
        per = {tk: q(t.tensor([float(x) for x, q_, t_ in zip(d, spl, task) if q_ == "dev" and t_ == tk]), .5) for tk in TASKS}
        P("  dev median dist per task: " + " ".join(f"{k[:5]} {v:.3f}" for k, v in per.items()))
        # modal cluster per task (dev)
        modal = {}
        for tk in TASKS:
            cc = Counter(a_ for a_, q_, t_ in zip(a, spl, task) if q_ == "dev" and t_ == tk)
            k0, v0 = cc.most_common(1)[0]
            modal[tk] = (k0, v0 / sum(cc.values()))
        summary[(c, l)] = dict(nmi=nm, modal=modal, wk_med=q(wd_v, .5), wk_p99=q(wd_v, .99),
                               dev_med=q(t.tensor([float(x) for x, q_ in zip(d, spl) if q_ == "dev"]), .5),
                               dev_out=float((t.tensor([float(x) for x, q_ in zip(d, spl) if q_ == "dev"]) > q(wd_v, .99)).float().mean()),
                               wk_share={k: wcnt[k] / len(wa_v) for k in range(8)})
P("\n== SUMMARY: per layer, dev modal cluster (share) per task, WikiText share of that cluster, NMI, distances")
for (c, l), s in summary.items():
    P(f"{c} L{l:2d} NMI {s['nmi']:.2f} | " + " ".join(f"{tk[:5]} c{k}({100*v:.0f}%,wk{100*s['wk_share'][k]:.0f}%)" for tk, (k, v) in s['modal'].items())
      + f" | dist med wk {s['wk_med']:.3f} dev {s['dev_med']:.3f}; dev beyond wk p99 {100*s['dev_out']:.0f}%")

# ------------------------------------------------------------- 4. qualitative (DEV)
P("\n== QUALITATIVE (dev): top-8 filtered tokens, concept best rank, router cluster (A)")
tok = model.tokenizer
ex = t.load(f"{R2}/A/A_full/experts_K8.pt", map_location="cpu", weights_only=True)
EX = {int(l): v.to(dev) for l, v in ex["experts"].items()}
del ex
FIX = {l: v.to(dev) for l, v in lensd(f"{R2}/A/A_full/jpp_A_full.pt").items()}
RS = {l: v.to(dev) for l, v in lensd(f"{R}/artifacts/baselines/jpp9b_prefit.pt").items()}


def hard(l, h):
    a, _ = route("A", l, h)
    out = t.stack([h @ EX[l][k].float().T for k in range(8)], 1)
    return out[t.arange(h.shape[0], device=dev), a], a


pick = [("association", "office-h"), ("association", "pregnant"), ("association", "art-h"),
        ("association", "churchill"), ("multilingual", "fr-color-night"), ("multilingual", "de-month-after-jan"),
        ("multihop", "louvre-language"), ("typo", "typo-already")]
key2i = {(b.eval_slugs[i], b.item_names[i]): i for i in range(b.num_items)}
for k in pick:
    assert part.get(k) == "dev", k
pi = [key2i[k] for k in pick]
cands = [b.candidates[i] for i in pi]
for l in (12, 20, 24, 28):
    h = b.resid[l][pi].to(dev)
    vh, a = hard(l, h)
    for name, v in (("fixed", h @ FIX[l].T), ("hard ", vh)):
        lg = unembed_logits(model, v, b.excluded_ids, chunk_size=16)
        rk = candidate_ranks(lg, cands)
        top = lg.topk(8, -1).indices.tolist()
        for r, (k, c) in enumerate(zip(pick, cands)):
            toks = " | ".join(repr(tok.decode([j]))[1:-1] for j in top[r])
            best = min(rk[r].values()) if rk[r] else None
            P(f"L{l} {name} {k[1]:18s} c{int(a[r])} concept {'/'.join(c)} rank {best}: {toks}")

# '@' share in top-10 at L24/L28, dev association, plus per-task rank changes
P("\n== '@' share of top-10 slots, dev association (fixed A_full vs hard A vs released J++)")
assoc = [i for i in idx if part[(b.eval_slugs[i], b.item_names[i])] == "dev" and b.eval_slugs[i] == "association"]
for l in (20, 24, 28):
    h = b.resid[l][assoc].to(dev)
    vh, a = hard(l, h)
    out = []
    for name, v in (("fixed", h @ FIX[l].T), ("hard", vh), ("released", h @ RS[l].T)):
        lg = unembed_logits(model, v, b.excluded_ids, chunk_size=16)
        top = lg.topk(10, -1).indices
        at = sum(tok.decode([j]).strip().startswith("@") for j in top.flatten().tolist()) / top.numel()
        out.append(f"{name} {100*at:.0f}%")
    P(f"L{l}: " + " ".join(out) + f" | clusters {dict(Counter(a.tolist()))}")
P("DONE")
