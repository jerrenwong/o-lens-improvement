"""test: a fake al3 meta whose 'new' sentences are 24 al2 sentences, to check label.py / collect.py against
sub2 al2_ranks.pt and al2 X_L*.pt / mp1 Xmean_L*.pt."""
import random, torch
m2 = torch.load("/workspace/o-lens-improvement/.hf27/al2/meta.pt", weights_only=False)
idx = sorted(random.Random(3).sample(range(len(m2["words"])), 24))
rows = [dict(word=m2["words"][i], style=m2["style"][i], sent=m2["sents"][i], src="new", al2_idx=-1, new_idx=j,
             split="train", type="THING") for j, i in enumerate(idx)]
torch.save(dict(rows=rows, new_sents=[m2["sents"][i] for i in idx], al2_rows=idx), "/root/al3/check/meta.pt")
print("ok", idx[:5])
