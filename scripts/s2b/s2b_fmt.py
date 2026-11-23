"""s2b label-free format check: for each unstated-word wording, the unmasked top-1 token of unembed(unit(Y_k))
(= the model's top first-answer token after the prefilled label); share that is not a word (no \\w char), share junk
(special/added or whitespace-only, the fit's junk rule), per wording and pooled; most frequent first tokens.
Usage: python s2b_fmt.py <model> <acts.pt>"""
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

torch.set_grad_enabled(False)
M, P = sys.argv[1], sys.argv[2]
model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
Y = torch.load(P)["Y_new"]
V = model.unembed(torch.zeros(1, model.d_model, device="cuda")).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
nonword = torch.tensor([not re.search(r"\w", s) for s in strs])
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
junk = torch.zeros(V, dtype=torch.bool)
junk[[i for i in special if i < V]] = True
junk |= torch.tensor([s.strip() == "" for s in strs])
allr = []
for k in range(Y.shape[0]):
    y = Y[k].float()
    y = y / y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    r = torch.cat([model.unembed(y[i:i + 250].cuda()).float().argmax(-1).cpu() for i in range(0, len(y), 250)])
    allr.append(r)
    print(f"{M} W{k}: N={len(r)} not-a-word {nonword[r].float().mean():.4f}  junk {junk[r].float().mean():.4f}  "
          f"top first tokens {Counter(strs[t] for t in r.tolist()).most_common(12)}", flush=True)
r = torch.cat(allr)
print(f"{M} POOLED: not-a-word {nonword[r].float().mean():.4f}  junk {junk[r].float().mean():.4f}  "
      f"top first tokens {Counter(strs[t] for t in r.tolist()).most_common(15)}", flush=True)
