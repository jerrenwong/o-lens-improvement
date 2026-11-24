"""Jacobian A-lens (w3_lib) under the evaluator's --filter level: at f3 the FIT ranks used to choose lam are computed over
token groups, exactly as auto/candidates/w1_alens.py does for the ridge A-lens. env W3_MODE = add | alone; W3_JT, W3_JU as w3_lib."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_lib  # noqa: E402
from fm27_lib import NEG, logits  # noqa: E402
from olens.readout import candidate_ranks, group_candidate_ranks  # noqa: E402


def build(ctx):
    gid = getattr(ctx.fit, "group_ids", None)
    fn = w3_lib.build_add if os.environ.get("W3_MODE", "add") == "add" else w3_lib.build_alone
    if gid is None:
        return fn(ctx)

    def ranks_g(ctx, lens, mask):
        rk_f, rk_u = {}, {}
        for l in ctx.readout_layers:
            L = logits(ctx.model, lens.transport(ctx.fit.resid[l], l))
            rk_u[l] = candidate_ranks(L, ctx.fit.candidates)
            rk_f[l] = group_candidate_ranks(L.masked_fill(mask, NEG), ctx.fit.candidates, gid)
        return rk_f, rk_u

    orig = w3_lib.ranks; w3_lib.ranks = ranks_g
    try:
        return fn(ctx)
    finally:
        w3_lib.ranks = orig
