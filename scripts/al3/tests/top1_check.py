"""diagnostic: most frequent top-1 tokens of maps a/b/c at L28 on T2 rows (CPU)."""
import json, collections, torch
R = "/workspace/o-lens-improvement"
torch.set_num_threads(16)
F = torch.load(f"{R}/.hf27/al3/fit_ranks.pt", weights_only=False)
T2 = F["sets"]["T2"]
N2 = 118892
X = torch.cat([torch.load(f"{R}/.hf27/al2/X_L28.pt"), torch.load(f"{R}/.hf27/al3/X_L28.pt")])[T2].float()
U = torch.load(f"{R}/.hf27/al1/U.pt").float()
g = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)["g"].float()
strs = torch.load(f"{R}/.hf27/al1/meta.pt", weights_only=False)["strs"]
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs])
for m in "abc":
    M = torch.load(f"{R}/.hf27/al3/M_{m}.pt")[28]
    v = (X - M["mx"]) @ (M["W"].float() * M["scale"]) + M["my"]
    lg = ((v * g) @ U.T).masked_fill(excl, float("-inf"))
    t1 = lg.argmax(-1).tolist()
    c = collections.Counter(strs[t] for t in t1)
    bias = ((M["my"] * g) @ U.T).masked_fill(excl, float("-inf")).topk(5).indices.tolist()
    print(m, "distinct top1", len(c), "of", len(t1), c.most_common(8), "| bias top5", [strs[t] for t in bias],
          "| |my| / mean|Wx|", float(M["my"].norm()), float((v - M["my"]).norm(dim=-1).mean()))
