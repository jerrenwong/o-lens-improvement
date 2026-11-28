"""q2 (T4b): the A-lens family with an arbitrary question set. Same lens object, lam grid and lam search as the frozen
family (alens.py via w1_alens.py / w9_alens.py: w9 map format), generalised from two maps to N:
    a(h) = unit( sum_i unit(map_i(h)) )         (mean of the unit-normalised map outputs, re-normalised)
    ALENS_BASE none -> a(h);  JPP -> unit(J++ h) + lam * a(h), lam on the FIT split (alens.GRID, alens rule).
Env: Q2_MAPS  comma-separated map files (w9 format: mx, W/scale, my); a bare stem is looked up in .hf27/q2/maps then
              .hf27/w9/maps; "<tag>" in a name is replaced by the model tag (9b / 2b).
     Q2_PRIOR 1 -> each map's output without its constant: (h - mx) W, i.e. map(h) minus its mean output over the
              training passages (mx = the training mean of X for these linear maps, so the mean output is exactly my).
     ALENS_BASE as alens.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alens  # noqa: E402
import w1_alens  # noqa: E402
from fm27_lib import unit  # noqa: E402
from w9_alens import emap_w9, load_maps_w9  # noqa: E402

R = "/workspace/o-lens-improvement"


def resolve(name, tag):
    name = name.strip().replace("<tag>", tag)
    if os.path.exists(name):
        return name
    for d in (f"{R}/.hf27/q2/maps", f"{R}/.hf27/w9/maps"):
        p = f"{d}/{name}" + ("" if name.endswith(".pt") else ".pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


class MA:
    def __init__(self, base, E, E2, lam):  # E = list of maps (E2 unused; kept for alens.build's call)
        self.base, self.E, self.lam = base, E, lam

    def transport(self, h, layer, resid_all=None):
        if self.base is not None and self.lam == 0:
            return unit(h @ self.base[layer].T.to(h))
        a = unit(sum(unit(emap_w9(m, h, layer)) for m in self.E))
        if self.base is None:
            return a
        return unit(h @ self.base[layer].T.to(h)) + self.lam * a


def build(ctx):
    tag = alens.TAG.get(ctx.model.d_model)
    paths = [resolve(x, tag) for x in os.environ["Q2_MAPS"].split(",") if x.strip()]
    maps = [load_maps_w9(p, ctx.device) for p in paths]
    if os.environ.get("Q2_PRIOR", "0") == "1":
        maps = [{l: (t[0], t[1], 0 * t[2]) + tuple(t[3:]) for l, t in m.items()} for m in maps]
    print(f"[q2_alens] {len(maps)} maps: {[Path(p).name for p in paths]} prior_removed={os.environ.get('Q2_PRIOR', '0')}",
          flush=True)
    alens.load_maps = lambda path, dev: maps
    alens.A = MA
    return w1_alens.build(ctx)
