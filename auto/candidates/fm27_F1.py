"""fm27 F1 rerank-only: the ask-map may only reorder J++'s filtered top-K (zJ + mu zE); (K, mu) chosen on FIT by
max pass@1 + pass@10 (ties -> smaller mu, then smaller K). Vector realisation: see fm27_lib.F1Lens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import KS, MUS, Base, F1Lens, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402


def build(ctx):
    mask = excl_mask(ctx)
    print(fmt("J++", fit_eval_lens(ctx, Base(ctx, 0.0), mask)), flush=True)
    res = []
    for mu in MUS:
        for K in KS:
            s = fit_eval_lens(ctx, F1Lens(ctx, K, mu), mask)
            print(fmt(f"F1v K{K} mu{mu}", s), flush=True)
            res.append(((K, mu), s))
    K, mu = choose(res)
    print(f"[fm27 F1] chosen K {K} mu {mu}", flush=True)
    return F1Lens(ctx, K, mu)
