"""q9: 'verbalisable-concept' output subspace for Qwen3.5-9B, label-free.
PCA of the centred elicitation targets (mean of the three unit-normalised final-layer answer states over 28.5k
ordinary-text prefixes; data from the e1 collection). Saves /root/e1/Ypcs_9b.pt = dict(V=[4096,4096] eigenvectors by
decreasing eigenvalue, evals, mean). Prints the spectrum and decoded top directions."""
import re, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model
torch.set_grad_enabled(False)
def unit(x): return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
A = torch.load("/root/e1/acts.pt")
Y = sum(unit(A["Y"][k].float()) for k in range(3)) / 3
Y = Y[2000:].cuda().double()                     # training rows only (val/test rows 0:2000 untouched)
mean = Y.mean(0); Yc = Y - mean
ev, V = torch.linalg.eigh(Yc.T @ Yc / len(Yc)); ev, V = ev.flip(0).clamp_min(0), V.flip(1)
cum = (ev.cumsum(0) / ev.sum()).cpu()
print("variance explained by top-k PCs:", {k: round(float(cum[k - 1]), 3) for k in (8, 32, 64, 128, 256, 512, 1024, 2048)})
print("participation ratio:", round(float(ev.sum() ** 2 / (ev ** 2).sum()), 1))
torch.save(dict(V=V.float().cpu(), evals=ev.float().cpu(), mean=mean.float().cpu()), "/root/e1/Ypcs_9b.pt")
model = get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa"); tok = model.tokenizer
Vn = model.unembed(torch.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
mask = torch.tensor([not re.search(r"\w", s) for s in strs]).cuda()
def dec(v): return " ".join(strs[j].strip() for j in model.unembed(v[None].float().cuda()).float().masked_fill(mask, float("-inf")).topk(7, -1).indices[0].tolist())
print("mean target decodes to:", dec(mean))
for k in range(6): print(f"PC{k+1}+ : {dec(V[:, k])}   |   PC{k+1}- : {dec(-V[:, k])}")
print("saved")
