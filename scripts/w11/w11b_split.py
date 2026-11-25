"""w11b: M1 sketch noise. Split-half cosine (even vs odd windows) of the per-token direction estimates, binned by
W_X and by n_eff = W^2 / sum p^2, for the token subset (fit+dev candidate ids + 3000 random ids).
    python scripts/w11/w11b_split.py 2b  -> prints a table, writes runs/w11/m1_2b_splithalf.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
tag = sys.argv[1]
sh = torch.load(R / f"runs/w11/m1_{tag}_splithalf.pt")
W = sh["W"].double(); W2 = sh["W2"].double()
neff = W ** 2 / W2.clamp_min(1e-300)
bins = {"W": (W, [0, 0.01, 0.1, 1, 10, 100, 1e3, 1e9]), "n_eff": (neff, [0, 3, 10, 30, 100, 300, 1e3, 1e9])}
out = {}
for var in ("Dpu", "Dhy", "Apu", "Ahy"):
    for l, c in sh[var].items():
        for bn, (x, edges) in bins.items():
            row = {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (x >= lo) & (x < hi)
                if k.sum():
                    row[f"{lo:g}-{hi:g}"] = (round(float(c[k].mean()), 3), int(k.sum()))
            out[f"{var} L{l} by {bn}"] = row
            print(f"{var} L{l} by {bn}: " + "  ".join(f"[{b}] {v[0]:+.3f} (n={v[1]})" for b, v in row.items()))
(R / f"runs/w11/m1_{tag}_splithalf.json").write_text(json.dumps(out, indent=1))
