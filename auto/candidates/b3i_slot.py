"""b3i: SLOT REUSE: slots 4 and 8 return E_a(resid_all[a]), E_b(resid_all[b]) for the two layers >= 12 with best label-free held-out agreement; other slots plain J++ (see b3i_lib)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3i_lib import build_maps, ELens  # noqa: E402


def build(ctx):
    maps, info = build_maps(ctx)
    L = ctx.readout_layers
    best2 = sorted([l for l in L if l >= 12], key=lambda l: -info[l])[:2]
    print('[b3i] slot sources', best2, flush=True)
    return ELens(ctx.JPP, maps, 'slot', slot_src={L[0]: best2[0], L[1]: best2[1]})
