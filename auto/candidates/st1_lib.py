"""st1 shared code (not a candidate). Lens L (fm27_F4 pattern: v = unit(JPP h) + lam unit(E h + b), lam chosen on
FIT from fm27_lib.GRID by max pass@1 + pass@10, ties -> smaller lam) with the ask-map swapped for an st1 map
(.hf27/st1/E_<tag>_<var>.pt; scripts/st1/st1_fit.py, st1_p3.py); var "B0" = the current map .hf27/ask3/E_<tag>_t3.pt.
ALONE: the map read alone, v = unit(E h + b) (no J++ term, no lam)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import GRID, Base, choose, excl_mask, fit_eval_lens, fmt, unit  # noqa: E402

R = "/workspace/o-lens-improvement"
TAGS = {2048: "2b", 4096: "9b", 5120: "27b"}


def map_path(ctx, var):
    tag = TAGS[ctx.model.d_model]
    return f"{R}/.hf27/ask3/E_{tag}_t3.pt" if var == "B0" else f"{R}/.hf27/st1/E_{tag}_{var}.pt"


class St1(Base):
    def __init__(self, ctx, var, lam=0.0, alone=False):
        self.JPP = ctx.JPP
        self.maps = load_maps(map_path(ctx, var), ctx.device)
        self.lam, self.centred, self.alone = lam, False, alone

    def transport(self, h, layer, resid_all=None):
        if self.alone:
            mx, W, my = self.maps[layer]
            return unit((h - mx) @ W + my)
        return super().transport(h, layer, resid_all)


def build_L(ctx, var):
    mask = excl_mask(ctx)
    res = []
    for lam in GRID:
        s = fit_eval_lens(ctx, St1(ctx, var, lam), mask)
        print(fmt(f"st1 {var} lam{lam}", s), flush=True)
        res.append((lam, s))
    lam = choose(res)
    print(f"[st1 {var}] chosen lam {lam}", flush=True)
    return St1(ctx, var, lam)


def build_alone(ctx, var):
    lens = St1(ctx, var, alone=True)
    print(fmt(f"st1 {var} ALONE", fit_eval_lens(ctx, lens, excl_mask(ctx))), flush=True)
    return lens


MID = {"2b": [9, 15], "9b": [12, 20], "27b": [24, 40]}


def qual(ctx, best):
    """FIT split only: 6 fit items whose concept J++ misses (no candidate in J++'s filtered top-10 at any readout
    layer), spread over tasks; top-8 tokens (filtered as scored) of B0 alone and `best` alone at two mid layers.
    Returns plain J++ (its dev score must reproduce the J++ run: sanity)."""
    import torch
    from olens.readout import unembed_logits
    tag = TAGS[ctx.model.d_model]
    tok, f = ctx.model.tokenizer, ctx.fit
    jpp = St1(ctx, "B0", 0.0)
    miss = torch.ones(len(f.eval_slugs), dtype=torch.bool)
    for l in ctx.readout_layers:
        lg = unembed_logits(ctx.model, jpp.transport(f.resid[l], l), f.excluded_ids)
        top = lg.topk(10, -1).indices.cpu()
        for p, c in enumerate(f.candidates):
            ids = torch.tensor([i for v in c.values() for i in v])
            if torch.isin(top[p], ids).any():
                miss[p] = False
    pick, per = [], {}
    for p in miss.nonzero()[:, 0].tolist():  # round-robin over tasks, first items in fit order
        per.setdefault(f.eval_slugs[p], []).append(p)
    while len(pick) < 6 and any(per.values()):
        for t in sorted(per):
            if per[t] and len(pick) < 6:
                pick.append(per[t].pop(0))
    print(f"[st1 qual] J++ misses {int(miss.sum())}/{len(miss)} fit items; showing {len(pick)}", flush=True)
    bests = [best] if isinstance(best, str) else list(best)
    lenses = {"B0": St1(ctx, "B0", alone=True), **{b: St1(ctx, b, alone=True) for b in bests}}
    for p in pick:
        cands = f.candidates[p]
        ids = {i for v in cands.values() for i in v}
        print(f"\n[qual] {f.eval_slugs[p]} / {f.item_names[p]}  concept(s): {list(cands)}", flush=True)
        for l in MID[tag]:
            h = f.resid[l][p:p + 1]
            for name, lens in lenses.items():
                lg = unembed_logits(ctx.model, lens.transport(h, l), f.excluded_ids)[0]
                top = lg.topk(8).indices.tolist()
                s = ", ".join(("*" if t in ids else "") + repr(tok.decode([t]).strip())[1:-1] for t in top)
                print(f"   L{l:2d} {name:8s} {s}", flush=True)
    return jpp


class Scaled:
    """J++ base of lens L (unit(J++ h), = any st1 lens at lam 0) times a constant s: mathematically the same lens;
    only bf16 rounding in unembed differs (noise-floor check)."""

    def __init__(self, ctx, s):
        self.JPP, self.s = ctx.JPP, s

    def transport(self, h, layer, resid_all=None):
        return self.s * unit(h @ self.JPP[layer].T)


class ScaledLens:
    """Any lens times a constant s (lam chosen on FIT unscaled, as in build_L): same lens mathematically;
    only bf16 rounding of the dev logits differs (noise-floor replicates)."""

    def __init__(self, lens, s):
        self.lens, self.s = lens, s

    def transport(self, h, layer, resid_all=None):
        return self.s * self.lens.transport(h, layer, resid_all)


def build_scaled(ctx, var, s):
    if var == "jppu":
        return Scaled(ctx, s)
    return ScaledLens(build_L(ctx, var), s)
