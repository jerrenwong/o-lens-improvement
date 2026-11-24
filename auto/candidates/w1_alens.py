"""w1: the frozen A-lens family (auto/candidates/alens.py, env ALENS_BASE) under the evaluator's --filter level.
f1 / f2: alens.build unchanged (its lambda search masks ctx.fit.excluded_ids, which the evaluator already set to the level).
f3: the same build, with the FIT ranks of the lambda search computed over token groups (ctx.fit.group_ids,
olens.readout.group_candidate_ranks) instead of tokens; unfiltered ranks unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alens  # noqa: E402
from fm27_lib import NEG, logits  # noqa: E402
from olens.readout import candidate_ranks, group_candidate_ranks  # noqa: E402


def build(ctx):
    gid = getattr(ctx.fit, "group_ids", None)
    print(f"[w1] filter {getattr(ctx.fit, 'filter_level', 'f1')}: {len(ctx.fit.excluded_ids)} ids masked, "
          f"{'no groups' if gid is None else str(int(gid.max()) + 1) + ' groups'}", flush=True)
    if gid is None:
        return alens.build(ctx)

    def ranks_g(ctx, lens, mask):  # same as ao1_lib.ranks, filtered ranks over groups
        rk_f, rk_u = {}, {}
        for l in ctx.readout_layers:
            L = logits(ctx.model, lens.transport(ctx.fit.resid[l], l))
            rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
            rk_f[l] = group_candidate_ranks(L.masked_fill(mask, NEG), ctx.fit.candidates, gid)
        return rk_f, rk_u

    orig = alens.ranks
    alens.ranks = ranks_g
    try:
        return alens.build(ctx)
    finally:
        alens.ranks = orig
