"""ex2: build a lens {layer: [F, N] fp32 on ctx.device} from a spec = a per-layer linear combination of stored
matrices. Sources: "E:<fit>:<e>" expert e of a fit, "J:<fit>" its pooled lens (exact fp32 file), "L:<fit>" its
combined J++ file, "RJPP" / "RJ" = ctx.JPP / ctx.J (the released J++ and the scorer's J-lens).
Loaded files are cached for the life of the process (sys.modules keeps this module across evaluate.main calls)."""
import sys

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
from ex2_common import FITS, LAYERS, load_experts, load_matrix_dict  # noqa: E402

_CACHE = {}


def _get(key):
    if key not in _CACHE:
        kind, fit = key
        m = FITS[fit]["m"]
        if kind == "E":
            _CACHE[key] = load_experts(fit)[0]
        elif kind == "J":
            _CACHE[key] = load_matrix_dict(FITS[fit]["pooled"], LAYERS[m])
        elif kind == "L":
            _CACHE[key] = load_matrix_dict(FITS[fit]["L"], LAYERS[m])
    return _CACHE[key]


def source(src, layer, ctx):
    p = src.split(":")
    if p[0] == "RJPP":
        return ctx.JPP[layer]
    if p[0] == "RJ":
        return ctx.J[layer]
    if p[0] == "E":
        return _get(("E", p[1]))[layer][int(p[2])]
    return _get((p[0], p[1]))[layer]


@torch.no_grad()
def build(spec, ctx):
    out = {}
    for layer in ctx.readout_layers:
        acc = None
        for src, coef in spec["layers"][str(layer)]:
            if coef == 0:
                continue
            x = source(src, layer, ctx).to(ctx.device, torch.float32) * float(coef)
            acc = x if acc is None else acc + x
        out[layer] = acc
    return out
