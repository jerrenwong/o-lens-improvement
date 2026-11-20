"""e4: does the distilled map's benchmark gain depend on the AMOUNT of distillation data?
Refit the frozen 9B recipe (same targets, same ridge grid/rule, same val set) on the first n training prefixes of a
fixed random order, n in {250, 1000, 4000, 16000, 28500}; save maps to /root/e1/maps_n<n>.pt (pack format of e2_fit).
Prints held-out agreement per n. Benchmark scoring is done by auto/candidates/e4_scale.py (env E4_MAP)."""
import json, re, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model
M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
torch.set_grad_enabled(False); dev = "cuda"
def unit(x): return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
A = torch.load("/root/e1/acts.pt"); N = A["Y"][0].shape[0]
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
mask = torch.tensor([(not re.search(r"\w", s)) or (s.strip().lower() in FUNC) for s in strs], device=dev)
norm_s = [s.strip().lower() for s in strs]; vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
def topk(v, k=10, bs=250):
    return torch.cat([model.unembed(v[i:i + bs].to(dev).float()).float().masked_fill(mask, float("-inf")).topk(k, -1).indices for i in range(0, v.shape[0], bs)])
def agree(top, word): return (NORM[top] == NORM[word][:, None]).any(-1)
Ymean = sum(unit(A["Y"][k].float()) for k in range(3)) / 3
va, te = slice(0, NV), slice(NV, NV + NT)
word = topk(Ymean, 1)[:, 0]
order = torch.randperm(N - NV - NT, generator=torch.Generator().manual_seed(7)) + NV + NT
SIZES = [250, 1000, 4000, 16000, N - NV - NT]
for n in SIZES:
    idx = order[:n]; maps = {}; anyE = torch.zeros(NT, dtype=torch.bool, device=dev); line = []
    for l in LAYERS:
        X = A["X"][l].float().to(dev); Xtr = X[idx.to(dev)]; Ytr = Ymean[idx].to(dev)
        mx = Xtr.mean(0); Xc = (Xtr - mx).double(); my = Ytr.mean(0)
        s, U = torch.linalg.eigh(Xc.T @ Xc); s = s.clamp_min(0); scale = s.mean().item()
        UtB = U.T @ (Xc.T @ (Ytr - my).double())
        best = None
        for c in GRID:
            W = (U @ (UtB / (s + c * scale)[:, None])).float()
            a = agree(topk((X[va] - mx) @ W + my), word[va]).float().mean().item()
            if best is None or a >= best[0]: best = (a, c, W)
        _, c, W = best
        h = agree(topk((X[te] - mx) @ W + my), word[te]); anyE |= h; line.append(f"L{l} {h.float().mean():.3f}")
        sc = W.abs().max().item()
        maps[l] = dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.cpu(), c=c)
        del X, Xc, U, UtB, W; torch.cuda.empty_cache()
    torch.save(maps, f"/root/e1/maps_n{n}.pt")
    print(f"n={n:6d} held-out top-10 agreement: {' '.join(line)}  any layer {anyE.float().mean():.3f}", flush=True)
print("saved")
