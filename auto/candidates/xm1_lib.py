"""xm1 shared code (not a candidate). PREREG_v5 lens L for the non-Qwen models, model-generic through ctx:
the model is identified by ctx.model.tokenizer.name_or_path (d_model alone is ambiguous: OLMo-3-7B and Qwen3.5-9B
are both 4096). Maps: .hf27/xm1/E_<tag>_t3.pt (scripts/xm1/collect.py + fit_map.py), shuffled control E_<tag>_t3_shuf.pt.
Same lens classes and selection rule as fm27_lib / ask3_lib / e2_lib (imported, not edited):
  L      = unit(J++ h) + lam unit(E h + b), lam = argmax over fm27_lib.GRID of FIT (pass@1 + pass@10), ties -> smaller
  jppu   = unit(J++ h)                       (L at lam 0)
  Lshuf  = unit(J++ h) + lam_L unit(E_shuf h + b_shuf)
  Lprior = unit(J++ h) + lam_L unit(b)       (constant-only: the main map's mean output, same vector for every prompt)
  alone  = unit(E h + b)
lam_L for the controls is recomputed with the identical FIT selection (deterministic; printed), not hard-coded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import ELens  # noqa: E402
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import GRID, Base, choose, excl_mask, fit_eval_lens, fmt, unit  # noqa: E402,F401

R = "/workspace/o-lens-improvement"
TAGS = {"allenai/Olmo-3-7B-Instruct": "olmo7b", "google/gemma-4-12B-it": "gemma12b", "google/gemma-4-E4B-it": "gemma4e4b"}


def tag_of(ctx):
    name = ctx.model.tokenizer.name_or_path
    if name not in TAGS:
        raise ValueError(f"xm1 lenses are for {list(TAGS)} only, got {name!r}")
    return TAGS[name]


def map_path(ctx, shuf=False):
    return f"{R}/.hf27/xm1/E_{tag_of(ctx)}_t3{'_shuf' if shuf else ''}.pt"


class XBase(Base):
    """fm27_lib.Base with the xm1 map of ctx's model."""

    def __init__(self, ctx, lam=0.0, shuf=False):
        self.JPP = ctx.JPP
        self.maps = load_maps(map_path(ctx, shuf), ctx.device)
        self.lam, self.centred = lam, False


class Alone(XBase):
    def transport(self, h, layer, resid_all=None):
        mx, W, my = self.maps[layer]
        return unit((h - mx) @ W + my)


def choose_lam(ctx, tag="xm1 L"):
    mask = excl_mask(ctx)
    res = []
    for lam in GRID:
        s = fit_eval_lens(ctx, XBase(ctx, lam), mask)
        print(fmt(f"{tag} lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[{tag}] chosen lam {lam} ({tag_of(ctx)})", flush=True)
    return lam


def prior_lens(ctx, lam):
    return ELens(ctx.JPP, load_maps(map_path(ctx), ctx.device), "prior", lam)
