"""W8 step 5 readout: lenses fitted on the organism itself, applied to the saved activations of gen_collect.py.
    python scripts/w8/read_refit.py <quirk>
Lenses: J = organism-refit J-lens (.hf27/w8/jlens_<tag>.pt, if present); A = organism-refit A-lens
(.hf27/w8/E_<quirk>_t3.pt / _s5.pt); JPPA = base J++ + 0.5 * refit A. Head = final norm + LM head only
(scripts/perf1/lite_model.py; untouched by the LoRA). Output .hf27/w8/tops_<quirk>_refit.pt."""
import os
import sys

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
from w8lib import BASE, BIG, LAYERS, R, LensSet, f4_filter, read_tops, tag_of  # noqa: E402

Q = sys.argv[1]
torch.set_grad_enabled(False)
sys.path.insert(0, "/root/jpp_lens/src")
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402

lens = get_lite_model(BASE)
V = lens._lm_head.weight.shape[0]
mask, gid, _, _ = f4_filter(lens.tokenizer, V)
jp = f"{BIG}/jlens_{tag_of(Q)}.pt"
names = ["A", "JPPA"] + (["J"] if os.path.exists(jp) else [])
LS = LensSet(J=jp if os.path.exists(jp) else None, E3=f"{BIG}/E_{Q}_t3.pt", E5=f"{BIG}/E_{Q}_s5.pt")
D = torch.load(f"{BIG}/acts_{Q}.pt")
tops = {}
for who in ("org", "base"):
    X = torch.cat(D[who])
    H = {l: X[:, li] for li, l in enumerate(LAYERS)}
    tops[who] = read_tops(lens, LS, names, H, mask, gid)
torch.save(dict(tops=tops, lenses=names, layers=LAYERS, jlens_refit=os.path.exists(jp)), f"{BIG}/tops_{Q}_refit.pt")
print("saved", names, flush=True)
