"""e3 diagnostic (CPU): how often the elicited top-1 content token (chosen family) is a 1-2 letter fragment."""
import sys
import torch
from transformers import AutoTokenizer
M, TAG = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(M)
T = torch.load(f"/workspace/o-lens-improvement/.hf27/e3/targets_{TAG}.pt")
w = T["TID"][:, 0].tolist()
s = [tok.decode([i]).strip() for i in w]
n = len(s)
print(f"{TAG} family {T['fam']}: elicited top-1 is 1 letter {sum(len(x) == 1 for x in s) / n:.3f}; "
      f"<=2 letters {sum(len(x) <= 2 for x in s) / n:.3f}; "
      f"target mass on <=2-letter tokens {sum(float(p) for row, prow in zip(T['TID'].tolist(), T['TP']) for i, p in zip(row, prow) if len(tok.decode([i]).strip()) <= 2) / n:.3f}")
