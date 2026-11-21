"""fm27 F3 top-1 protection (strict version): C (registered lam) with J++'s own filtered top-1 pushed to rank 1,
strictly above every other token (smallest push along that token's unembedding direction). No free parameter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import LAM_C, Base, excl_mask, fit_eval_lens, fmt  # noqa: E402
from fm27_push import F3s  # noqa: E402


def build(ctx):
    mask = excl_mask(ctx)
    lam = LAM_C[ctx.model.d_model]
    print(fmt("J++", fit_eval_lens(ctx, Base(ctx, 0.0), mask)), flush=True)
    print(fmt("C", fit_eval_lens(ctx, Base(ctx, lam), mask)), flush=True)
    lens = F3s(ctx, lam)
    print(fmt("F3s v", fit_eval_lens(ctx, lens, mask)), flush=True)
    print(f"[fm27 F3s] lam {lam}; rows not fixable on FIT: {lens.fallback}", flush=True)
    return F3s(ctx, lam)
