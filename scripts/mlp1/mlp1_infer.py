"""mlp1: inferable train rows from .hf27/sub2/al2_ranks.pt (labelled & rank <= 10) -> .hf27/mlp1/infer_rows.pt"""
import torch
B = "/workspace/o-lens-improvement/.hf27"
d = torch.load(f"{B}/sub2/al2_ranks.pt", weights_only=False)
print("note:", d["note"])
meta = torch.load(f"{B}/al2/meta.pt", weights_only=False)
tr = torch.tensor([s == "train" for s in meta["split"]])
ok = d["labelled"] & (d["rank"] <= 10)
print(f"labelled {int(d['labelled'].sum())}; inferable {int(ok.sum())}; train labelled {int((tr & d['labelled']).sum())}, "
      f"train inferable {int((tr & ok).sum())}; shuf<=10 {int((d['labelled'] & (d['rank_shuf'] <= 10)).sum())}")
torch.save((tr & ok).nonzero()[:, 0], f"{B}/mlp1/infer_rows.pt")
