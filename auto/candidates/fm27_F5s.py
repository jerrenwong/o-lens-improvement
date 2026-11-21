"""fm27 F5 top-1 from J++'s top-10 (strict): C (registered lam), but rank 1 may only go to a token J++ itself ranks
in its filtered top-10 (the one with the highest C logit), pushed strictly to rank 1. No free parameter.
Motivation (fm27 diagnosis, FIT): on 27B C's top-1 lies outside J++'s top-10 in 11-34% of rows per layer (9B 0-5%),
and those map-only winners (' **', ' language', passage-topic words) displace J++'s correct top-1, while C's
rank-1 gains come from concepts J++ already had at rank 2-4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import LAM_C, Base, excl_mask, fit_eval_lens, fmt  # noqa: E402
from fm27_push import F5s  # noqa: E402


def build(ctx):
    mask = excl_mask(ctx)
    lam = LAM_C[ctx.model.d_model]
    print(fmt("C", fit_eval_lens(ctx, Base(ctx, lam), mask)), flush=True)
    lens = F5s(ctx, lam)
    print(fmt("F5s v", fit_eval_lens(ctx, lens, mask)), flush=True)
    print(f"[fm27 F5s] lam {lam}; rows not fixable on FIT: {lens.fallback}", flush=True)
    return F5s(ctx, lam)
