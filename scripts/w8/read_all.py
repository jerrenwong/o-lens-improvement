"""W8 follow-up readout: base-fitted and organism-fitted lenses side by side on the saved activations (organism and
base control on identical tokens) of gen_collect.py.
    python scripts/w8/read_all.py <quirk>
Base-fitted: J, JPP, A, JA (= unit(J h) + 0.5 A), JPPA.  Organism-fitted (when present): rJ (.hf27/w8/jlens_<tag>.pt),
rA (.hf27/w8/E_<quirk>_t3/s5.pt), rJA, rJPP (.hf27/w8/jpp_<tag>.pt), rJPPA (= unit(rJPP h) + 0.5 rA).
Head = final norm + LM head (scripts/perf1/lite_model.py). Output .hf27/w8/tops_<quirk>_all.pt."""
import os
import sys

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
from w8lib import BASE, BIG, LAM, LAYERS, R, emap, f4_filter, load_jlens, load_maps, read_tops, tag_of, unit  # noqa: E402

Q = sys.argv[1]
torch.set_grad_enabled(False)
sys.path.insert(0, "/root/jpp_lens/src")
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402


class Lenses:
    def __init__(self):
        self.M = {"J": load_jlens("artifacts/baselines/jlens27b_prefit.pt"),
                  "JPP": load_jlens("artifacts/baselines/jpp27b_released.pt")}
        self.E = {"": (load_maps(f"{R}/.hf27/ask3/E_27b_t3.pt"), load_maps(f"{R}/.hf27/ask5/E_27b_s5.pt"))}
        self.names = ["J", "JPP", "A", "JA", "JPPA"]
        jp, pp = f"{BIG}/jlens_{tag_of(Q)}.pt", f"{BIG}/jpp_{tag_of(Q)}.pt"
        e3, e5 = f"{BIG}/E_{Q}_t3.pt", f"{BIG}/E_{Q}_s5.pt"
        if os.path.exists(jp):
            self.M["rJ"] = load_jlens(jp); self.names.append("rJ")
        if os.path.exists(pp):
            self.M["rJPP"] = load_jlens(pp); self.names.append("rJPP")
        if os.path.exists(e3) and os.path.exists(e5):
            self.E["r"] = (load_maps(e3), load_maps(e5)); self.names.append("rA")
            if "rJ" in self.M:
                self.names.append("rJA")
            if "rJPP" in self.M:
                self.names.append("rJPPA")

    def a(self, pre, h, l):
        E3, E5 = self.E[pre]
        return unit(unit(emap(E3, h, l)) + unit(emap(E5, h, l)))

    def apply(self, name, h, l):
        pre = "r" if name.startswith("r") else ""
        core = name[1:] if pre else name
        if core in ("J", "JPP"):
            v = h @ self.M[pre + core][l].T
            return v if core == "J" else unit(v)
        if core == "A":
            return self.a(pre, h, l)
        if core == "JA":
            return unit(h @ self.M[pre + "J"][l].T) + LAM * self.a(pre, h, l)
        if core == "JPPA":
            return unit(h @ self.M[pre + "JPP"][l].T) + LAM * self.a(pre, h, l)
        raise KeyError(name)


lens = get_lite_model(BASE)
V = lens._lm_head.weight.shape[0]
mask, gid, _, _ = f4_filter(lens.tokenizer, V)
LS = Lenses()
print("lenses", LS.names, flush=True)
D = torch.load(f"{BIG}/acts_{Q}.pt")
tops = {}
for who in ("org", "base"):
    X = torch.cat(D[who])
    H = {l: X[:, li] for li, l in enumerate(LAYERS)}
    tops[who] = read_tops(lens, LS, LS.names, H, mask, gid)
torch.save(dict(tops=tops, lenses=LS.names, layers=LAYERS), f"{BIG}/tops_{Q}_all.pt")
print("saved", LS.names, flush=True)
