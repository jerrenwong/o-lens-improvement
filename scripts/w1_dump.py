"""w1: dump the g1 kept sentences (words, styles, sents) from .hf27/g1/acts.pt to runs/w1/g1_kept.json."""
import json, torch
R = "/workspace/o-lens-improvement"
a = torch.load(f"{R}/.hf27/g1/acts.pt", map_location="cpu", weights_only=False, mmap=True)
out = [dict(word=w, style=int(s), sent=x) for w, s, x in zip(a["words"], a["styles"], a["sents"])]
with open(f"{R}/runs/w1/g1_kept.json", "w") as f:
    f.write(json.dumps(out))
print("kept", len(out))
