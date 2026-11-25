"""merge per-layer maps .hf27/w12/<tag>/maps/<cfg>_L<l>.pt -> <cfg>.pt {layer: {M, tau}}. Usage: w12_merge.py <tag> cfg1,cfg2,.."""
import os
import sys

import torch

R = "/workspace/o-lens-improvement"
tag, cfgs = sys.argv[1], sys.argv[2].split(",")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[tag]
D = f"{R}/.hf27/w12/{tag}/maps"
for c in cfgs:
    fs = {l: f"{D}/{c}_L{l}.pt" for l in LAYERS}
    miss = [l for l, f in fs.items() if not os.path.exists(f)]
    if miss:
        print(f"{c}: missing layers {miss}"); continue
    torch.save({l: torch.load(f) for l, f in fs.items()}, f"{D}/{c}.pt")
    print(f"{c}: merged")
