"""w9: the frozen A-lens family (alens.py via w1_alens.py: same lens object, same lam grid and lam search under the
evaluator's --filter level) with w9 maps, which may be one-hidden-layer MLPs:
    map(h) = (h - mx) W + my + W2 relu(((h - mx) / sx) W1^T + c1)      (linear maps: the first two terms only)
Map files (scripts/w9/w9_fit.py): per layer dict mx, W (/scale), scale, my [, W1, c1, W2, sx]; env ALENS_E (topic) /
ALENS_E2 (unstated word), ALENS_BASE none|JPP|JPP0 as in alens.py. Only alens.load_maps / alens.emap are swapped."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alens  # noqa: E402
import w1_alens  # noqa: E402


def load_maps_w9(path, dev):
    m = torch.load(path, map_location="cpu")
    out = {}
    for l, v in m.items():
        t = (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
        if v.get("W1") is not None:
            t = t + (v["W1"].to(dev).float(), v["c1"].to(dev).float(), v["W2"].to(dev).float(), v["sx"].to(dev).float())
        out[l] = t
    return out


def emap_w9(maps, h, l):
    m = maps[l]
    x = h - m[0]
    out = x @ m[1] + m[2]
    if len(m) > 3:
        W1, c1, W2, sx = m[3:]
        out = out + torch.relu((x / sx) @ W1.T + c1) @ W2.T
    return out


def build(ctx):
    alens.load_maps = load_maps_w9
    alens.emap = emap_w9
    return w1_alens.build(ctx)
