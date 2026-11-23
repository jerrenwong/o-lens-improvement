"""tl1 step 1 (PREREG_v4 Addendum F): T-lens maps = the A-lens ridge recipe with the regression INPUT replaced by the
passage's mean input embedding (no forward pass). Everything that makes the maps is copied unchanged from the A-lens
fits (scripts/ask3/ask3_fit9.py, scripts/ask5/ask5_fit.py, scripts/m2b/m2b_fit.py, scripts/s2b/s2b_fit.py,
scripts/xm1/fit_map.py): same passages (runs/e1/prefixes.json), same stored targets (answer states per passage and
phrasing), junk rule (format vector dropped when its UNMASKED top-1 is special/added/whitespace; target = mean of
kept unit vectors; passage dropped if none kept), elicited word = masked top-1 of unembed(target), splits (val
[0,1000), test [1000,2000), train rest), Ridge class, c grid, validation rule (the c with the largest val top-10 word
agreement, ties -> larger c), packing. ONE change: X = mean input embedding of the passage tokens (Qwen: the stored
ids, as the Qwen collections fed them; Olmo/Gemma: tok(text).input_ids with the BOS-forced harness tokenizer, as
scripts/xm1/collect.py), one map per question instead of one per layer; the same map is saved under every readout
layer key so the A-lens loader (load_maps / emap) works unchanged.
Label-free held-out diagnostics (test passages): T-map vs the existing A-maps (per layer) top-10 / top-1 agreement
with the model's elicited answer word, shuffled-target control, prior-only baselines.
Outputs .hf27/tl1/T_<tag>_t3.pt, T_<tag>_s5.pt, runs/tl1/fit_<tag>.json.
    python scripts/tl1/tl1_fit.py <tag>        (tag in 9b 2b 27b olmo7b gemma4e4b)"""
import json
import re
import sys
from collections import Counter

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/xm1")
sys.path.insert(0, f"{R}/scripts/tl1")
sys.path.insert(0, f"{R}/auto")
from evaluate_t import MODELS  # noqa: E402
from lite_model_x import get_lite_model  # noqa: E402
from tl1_lib import load_embedding  # noqa: E402

TAG = sys.argv[1]
H = f"{R}/.hf27"
CFG = {
    "9b": dict(model="Qwen/Qwen3.5-9B", qwen=True, X=f"{H}/ask2/acts_9b.pt",
               t3=(f"{H}/ask3/acts_9b.pt", f"{H}/ask3/E_9b_t3.pt"), s5=(f"{H}/ask5/acts_9b.pt", f"{H}/ask5/E_9b_s5.pt")),
    "2b": dict(model="Qwen/Qwen3.5-2B", qwen=True, X="/root/e1/acts_2b.pt",
               t3=(f"{H}/ask3/acts_2b.pt", f"{H}/ask3/E_2b_t3.pt"), s5=(f"{H}/ask5/acts_2b.pt", f"{H}/ask5/E_2b_s5.pt")),
    "27b": dict(model="Qwen/Qwen3.6-27B", qwen=True, X=f"{H}/m27/acts_27b.pt",
                t3=(f"{H}/ask3/acts_27b.pt", f"{H}/ask3/E_27b_t3.pt"), s5=(f"{H}/ask5/acts_27b.pt", f"{H}/ask5/E_27b_s5.pt")),
    "olmo7b": dict(model="allenai/Olmo-3-7B-Instruct", qwen=False, X=f"{H}/xm1/X_olmo7b.pt",
                   t3=(f"{H}/xm1/acts_olmo7b.pt", f"{H}/xm1/E_olmo7b_t3.pt"),
                   s5=(f"{H}/xm1/acts_olmo7b_s5.pt", f"{H}/xm1/E_olmo7b_s5.pt")),
    "gemma4e4b": dict(model="google/gemma-4-E4B-it", qwen=False, X=f"{H}/xm1/X_gemma4e4b.pt",
                      t3=(f"{H}/xm1/acts_gemma4e4b.pt", f"{H}/xm1/E_gemma4e4b_t3.pt"),
                      s5=(f"{H}/xm1/acts_gemma4e4b_s5.pt", f"{H}/xm1/E_gemma4e4b_s5.pt")),
}[TAG]
M = CFG["model"]
LAYERS = MODELS[M]["layers"]
assert MODELS[M]["tag"] == TAG
OUT = f"{H}/tl1"
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
torch.set_grad_enabled(False)
dev = "cuda"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def tload(p):
    for a in range(4):
        try:
            return torch.load(p, map_location="cpu")
        except Exception as e:  # noqa: BLE001  network fs
            print("load retry", p, repr(e)[:200], flush=True)
    return torch.load(p, map_location="cpu")


data = json.load(open(f"{R}/runs/e1/prefixes.json"))
N = len(data)
model = None
for attempt in range(4):
    try:
        model = get_lite_model(M)
        break
    except Exception as e:  # noqa: BLE001
        print("lite load retry", attempt, repr(e)[:300], flush=True)
tok = model.tokenizer

# ---- the ONE change: input features = mean input embedding of the passage tokens
EMB = load_embedding(M).to(dev)
if CFG["qwen"]:
    seqs = [o["ids"] for o in data]
else:
    seqs = [tok(o["text"]).input_ids for o in data]
print("passage tokens example:", [tok.decode([i]) for i in seqs[0][:6]], "... len", len(seqs[0]),
      f"; mean len {sum(map(len, seqs)) / N:.1f}", flush=True)
F = torch.stack([EMB[torch.tensor(s, device=dev)].float().mean(0) for s in seqs]).cpu()
del EMB
print(f"features {tuple(F.shape)}", flush=True)

V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
mask = nonword | func
norm_s = [s.strip().lower() for s in strs]
vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
special_ids = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special_ids |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
junk_tok[[i for i in special_ids if i < V]] = True
junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)
print(f"junk vocabulary: {int(junk_tok.sum())} ids", flush=True)


def topk(v, k=10, bs=250, masked=True):
    out = []
    for i in range(0, v.shape[0], bs):
        lg = model.unembed(v[i:i + bs].to(dev).float()).float()
        if masked:
            lg = lg.masked_fill(mask, float("-inf"))
        out.append(lg.topk(k, -1).indices)
    return torch.cat(out)


def agree(top, word):
    return (NORM[top] == NORM[word][:, None]).any(-1)


def targets(Y):
    Yk = [unit(Y[k].float()) for k in range(3)]
    raw1 = [topk(y, 1, masked=False)[:, 0] for y in Yk]
    junk = torch.stack([junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
    return T, cnt > 0, topk(T, 10)[:, 0], int(junk.sum())


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


def load_maps(path):
    m = tload(path)
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


XA = tload(CFG["X"])
X_all = XA["X"] if "X" in XA else XA
del XA
report = dict(tag=TAG, model=M, n_passages=N, feature_dim=F.shape[1], q={})
for Q in ("t3", "s5"):
    ypath, apath = CFG[Q]
    Y = tload(ypath)["Y_new"]
    T, valid, word, nj = targets(Y)
    del Y
    tr_m = valid.clone(); tr_m[:NV + NT] = False
    va_m = valid.clone(); va_m[NV:] = False
    te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
    tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].cpu() for m in (tr_m, va_m, te_m))
    wt = word[te_i.to(dev)]
    print(f"\n==== {Q}: dropped format-vectors {nj}; train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}", flush=True)
    prior_a = agree(topk(T[tr_i].mean(0, keepdim=True)).expand(len(te_i), -1), wt).float().mean().item()
    freq = torch.tensor([t for t, _ in Counter(word[tr_i.to(dev)].tolist()).most_common(10)], device=dev)[None].expand(len(te_i), -1)
    freq_a = agree(freq, wt).float().mean().item()
    # T map (one map, input = mean input embedding)
    Fd = F.to(dev)
    r = Ridge(Fd[tr_i.to(dev)])
    r.setY(T[tr_i].to(dev))
    sc = [agree(topk(r.predict(Fd[va_i.to(dev)], r.W(c_))), word[va_i.to(dev)]).float().mean().item() for c_ in GRID]
    b = max(range(len(GRID)), key=lambda i: (sc[i], i))
    c = GRID[b]
    W = r.W(c)
    tT = topk(r.predict(Fd[te_i.to(dev)], W))
    hT10, hT1 = agree(tT, wt), agree(tT[:, :1], wt)
    P = pack(r, W, c)
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(tr_i), generator=g)
    r.setY(T[tr_i][perm].to(dev))
    hS10 = agree(topk(r.predict(Fd[te_i.to(dev)], r.W(c))), wt).float().mean().item()
    del r, Fd
    torch.cuda.empty_cache()
    print(f"T map c={c:g} val@10 by c: " + " ".join(f"{a:g}:{s:.3f}" for a, s in zip(GRID, sc)), flush=True)
    with open(f"{OUT}/T_{TAG}_{Q}.pt", "wb") as f:  # same map under every readout layer (load_maps format)
        torch.save({l: P for l in LAYERS}, f)
    # A maps (existing, per layer) on the same test passages, for comparison
    A = load_maps(apath)
    arows, anyA = [], torch.zeros(len(te_i), dtype=torch.bool, device=dev)
    for l in LAYERS:
        mx, Wa, my = A[l]
        tA = topk((X_all[l][te_i].to(dev).float() - mx) @ Wa + my)
        hA10, hA1 = agree(tA, wt), agree(tA[:, :1], wt)
        anyA |= hA10
        both = (hA10 & hT10).float().mean().item()
        arows.append(dict(layer=l, A10=hA10.float().mean().item(), A1=hA1.float().mean().item(), A10_and_T10=both,
                          top1_same=(NORM[tA[:, 0]] == NORM[tT[:, 0]]).float().mean().item()))
    del A
    q = dict(c=c, val10_by_c=dict(zip(map(str, GRID), sc)), n_train=len(tr_i), n_val=len(va_i), n_test=len(te_i),
             T10=hT10.float().mean().item(), T1=hT1.float().mean().item(), shuf10=hS10, prior_top10=prior_a,
             freq_top10=freq_a, A_layers=arows, A10_best=max(a["A10"] for a in arows),
             A1_best=max(a["A1"] for a in arows), A10_any_layer=anyA.float().mean().item(), dropped_vectors=nj)
    report["q"][Q] = q
    print(f"held-out agreement with the elicited word ({Q}, n={len(te_i)}): T top10 {q['T10']:.3f} top1 {q['T1']:.3f} | "
          f"shuffled {hS10:.3f} | prior {prior_a:.3f} freq10 {freq_a:.3f}")
    print("layer | A top10 | A top1 | A&T top10 | same top-1 word")
    for a in arows:
        print(f"L{a['layer']:2d} | {a['A10']:.3f} | {a['A1']:.3f} | {a['A10_and_T10']:.3f} | {a['top1_same']:.3f}")
    print(f"A best layer top10 {q['A10_best']:.3f} top1 {q['A1_best']:.3f}; any layer {q['A10_any_layer']:.3f}", flush=True)
with open(f"{R}/runs/tl1/fit_{TAG}.json", "w") as f:
    json.dump(report, f, indent=1)
print("saved", flush=True)
