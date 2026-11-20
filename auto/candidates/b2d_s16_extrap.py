"""b2d idea 16: unit(JPP h) + g * (unit(JPP h) - unit(J h)); g chosen on the FIT split from
{-0.5, -0.25, 0, 0.25, 0.5, 1} (g=0 is J++; kept unless another g beats it on fit)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2d_common", Path(__file__).with_name("b2d_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

GS = [0.0, -0.5, -0.25, 0.25, 0.5, 1.0]


class Lens:
    def __init__(self, JPP, J, g):
        self.JPP, self.J, self.g = JPP, J, g

    def transport(self, h, layer, resid_all=None):
        P = C.unit(h @ self.JPP[layer].T)
        return P + self.g * (P - C.unit(h @ self.J[layer].T))


def build(ctx):
    best = None
    for g in GS:
        lens = Lens(ctx.JPP, ctx.J, g)
        sc = C.report(ctx, f"extrap g={g}", {l: lens.transport(ctx.fit.resid[l], l) for l in ctx.readout_layers})
        if best is None or sc > best[0]:
            best = (sc, g)
    print(f"[fit] chosen g={best[1]}", flush=True)
    return Lens(ctx.JPP, ctx.J, best[1])
