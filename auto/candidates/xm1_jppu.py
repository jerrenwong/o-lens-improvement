"""xm1 reference: unit-scaled J++, unit(J++ h) (= lens L at lam 0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xm1_lib import unit  # noqa: E402


class Lens:
    def __init__(self, ctx):
        self.JPP = ctx.JPP

    def transport(self, h, layer, resid_all=None):
        return unit(h @ self.JPP[layer].T)


def build(ctx):
    return Lens(ctx)
