import torch
R = "/workspace/o-lens-improvement"
m = torch.load("/root/al3/check/meta.pt", weights_only=False)
idx = torch.tensor(m["al2_rows"])
L2 = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt"); L3 = torch.load("/root/al3/check/labels_new.pt")
n = int(L3["labelled"].sum()); idx = idx[:n]; L3 = {k: v[:n] for k, v in L3.items() if torch.is_tensor(v)}
print("rank equal", int((L2["rank"][idx] == L3["rank"]).sum()), "of", len(idx), list(zip(L2["rank"][idx].tolist(), L3["rank"].tolist()))[:12])
print("rank_shuf equal", int((L2["rank_shuf"][idx] == L3["rank_shuf"]).sum()), "top3 equal", int((L2["top3"][idx] == L3["top3"]).all(1).sum()))
for l in (4, 28):
    a = torch.load(f"{R}/.hf27/al2/X_L{l}.pt")[idx].float(); b = torch.load(f"/root/al3/check/X_L{l}.pt").float()
    am = torch.load(f"{R}/.hf27/mp1/Xmean_L{l}.pt")[idx].float(); bm = torch.load(f"/root/al3/check/Xmean_L{l}.pt").float()
    cs = torch.cosine_similarity(a, b, dim=-1); cm = torch.cosine_similarity(am, bm, dim=-1)
    print(f"L{l} last cos min {cs.min():.4f} mean {cs.mean():.4f} | mean cos min {cm.min():.4f} mean {cm.mean():.4f}")
