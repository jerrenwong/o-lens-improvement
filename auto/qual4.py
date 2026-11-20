"""Qualitative follow-ups on lens failures (FIT + DEV items only; test never loaded into any statistic).
A. Right concept, wrong language: does the lens name the concept in Chinese?
B. Where does the '@' / sports-team junk in J++ L24/L28 come from (is it a sentence-final-period thing)?
C. Does the model name the association concept when asked with the full prompt?  Poetry: rhyme word at prompt end."""
import json, collections, statistics
from pathlib import Path
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
from olens.train import wikitext_windows
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]
J = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
LENS = {"J": J, "JPP": P}
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
def is_cjk(s): return any('一' <= ch <= '鿿' or '぀' <= ch <= 'ヿ' for ch in s)
cjk = t.tensor([is_cjk(s) for s in strs]).cuda()
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
print(f"vocab {Vn}; CJK tokens {int(cjk.sum())}; items fit+dev {len(idx)}", flush=True)

def macro(d, k, split):
    item = collections.defaultdict(list)
    for (i, key), r in d.items():
        if part[(b.eval_slugs[i], b.item_names[i])] == split: item[i].append(r <= k)
    ev = collections.defaultdict(list)
    for i, hs in item.items(): ev[b.eval_slugs[i]].append(sum(hs) / len(hs))
    return sum(sum(v) / len(v) for v in ev.values()) / len(ev), {e[:6]: round(sum(v) / len(v), 3) for e, v in sorted(ev.items())}

at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()
task_rows = collections.defaultdict(list)
for r, i in enumerate(idx): task_rows[b.eval_slugs[i]].append(r)
split_of = [part[(b.eval_slugs[i], b.item_names[i])] for i in idx]
fit_rows = [r for r in range(len(idx)) if split_of[r] == "fit"]
def dec(v, n=6): return " ".join(strs[j].strip() for j in model.unembed(v[None]).float().masked_fill(punct, float("-inf")).topk(n, -1).indices[0].tolist())
SV = {}
print("== 1. what the lens matrix itself prefers to say: top singular output directions, decoded (both signs)")
for n in LENS:
    for L in LAYERS:
        U, S, Vh = t.linalg.svd(LENS[n][L].cuda().float(), full_matrices=False); SV[(n, L)] = (U, S, Vh)
        h = b.resid[L][idx].cuda().float(); v = h @ LENS[n][L].cuda().float().T
        co = (v @ U[:, :20]) ** 2 / (v ** 2).sum(-1, keepdim=True)            # share of readout energy per top direction
        sh = {e[:5]: f"{float(co[rows, :1].sum(-1).mean()):.2f}/{float(co[rows, :5].sum(-1).mean()):.2f}/{float(co[rows].sum(-1).mean()):.2f}" for e, rows in sorted(task_rows.items())}
        print(f" {n:3s} L{L:2d} sigma1..5/median = {[round(float(x / S.median()), 1) for x in S[:5]]}  readout energy in top 1/5/20 dirs by task: {sh}")
        if L in (8, 20, 24, 28):
            for k in range(3):
                print(f"        u{k+1}+ : {dec(U[:, k])}    u{k+1}- : {dec(-U[:, k])}")
vm = (b.resid[24][[idx[r] for r in task_rows['association']]].cuda().float() @ P[24].cuda().float().T).mean(0)
print("   cos(mean J++ L24 association readout, u_k) k=1..8:", [round(float(t.cosine_similarity(vm, SV[('JPP', 24)][0][:, k], 0)), 2) for k in range(8)])
def lens_best(name, edit=None, mats=None):
    best = {}
    for L in LAYERS:
        Mx = (mats[(name, L)] if mats else LENS[name][L].cuda().float())
        v = b.resid[L][idx].cuda().float() @ Mx.T
        if edit is not None: v = edit(name, L, v)
        for s in range(0, len(idx), 32):
            lg = model.unembed(v[s:s+32]).float().masked_fill(punct, float("-inf"))
            for k in range(lg.shape[0]):
                i = idx[s + k]
                for key, ids in b.candidates[i].items():
                    best[(i, key)] = min(best.get((i, key), 10**9), int((lg[k] > lg[k, ids].max()).sum()) + 1)
    return best
def report(label, **kw):
    out = {}
    for n in LENS:
        bd = lens_best(n, **kw); out[n] = bd
        for s_ in ("fit", "dev"):
            p10, per = macro(bd, 10, s_); p1, _ = macro(bd, 1, s_)
            print(f"  {n:3s} {s_} {label:52s} {p10:.4f} / {p1:.4f}  {per}", flush=True)
    return out
print("\n== 2. remove the top-k singular directions from the lens matrix (data-free)")
report("plain")
for layers, ll in ((None, "all layers"), ((24, 28), "L24+L28 only")):
    for k in (1, 2, 3, 5, 10, 20):
        mats = {}
        for (n, L), (U, S, Vh) in SV.items():
            Mx = LENS[n][L].cuda().float()
            mats[(n, L)] = Mx - (U[:, :k] * S[:k]) @ Vh[:k] if (layers is None or L in layers) else Mx
        report(f"minus top-{k} singular directions, {ll}", mats=mats)
print("\n== 3. gated junk removal: remove direction u (unit mean of FIT association readouts) only when cos(readout, u) > tau")
def gated(tau, layers):
    cache = {}
    def edit(name, L, v):
        if layers is not None and L not in layers: return v
        if (name, L) not in cache:
            X = b.resid[L][[idx[r] for r in fit_rows if b.eval_slugs[idx[r]] == "association"]].cuda().float() @ LENS[name][L].cuda().float().T
            u = X.mean(0); cache[(name, L)] = u / u.norm()
        u = cache[(name, L)]
        c = t.cosine_similarity(v, u[None], -1)
        return t.where((c > tau)[:, None], v - (v @ u)[:, None] * u[None], v)
    return edit
for tau in (0.7, 0.8, 0.9):
    for layers, ll in ((None, "all layers"), ((24, 28), "L24+L28"), ((20, 24, 28), "L20-28")):
        report(f"gated tau={tau}, {ll}", edit=gated(tau, layers))
print("\n   share of rows above tau=0.8 by task and layer (J++)")
ed = gated(0.8, None)
for L in LAYERS:
    v = b.resid[L][idx].cuda().float() @ P[L].cuda().float().T; ed("JPP", L, v)
    X = b.resid[L][[idx[r] for r in fit_rows if b.eval_slugs[idx[r]] == "association"]].cuda().float() @ P[L].cuda().float().T
    u = X.mean(0); u = u / u.norm(); c = t.cosine_similarity(v, u[None], -1)
    print(f"   L{L}: " + "  ".join(f"{e[:6]} {float((c[rows] > 0.8).float().mean()):.2f}" for e, rows in sorted(task_rows.items())))
