"""m2b C_shuf: shuffled-target control for m2b_L on Qwen3.5-2B: J++ + lam * unit(shuffled-target map), map
.hf27/ask3/E_2b_t3_shuf.pt (same ridge c per layer, targets permuted over train prefixes). lam = 0.5 HARD-CODED:
it is the weight m2b_L chose on the 2B FIT split (seen in its dev run, auto/runs/m2b_L_2b, before this file was
written). PREREG_v4.md Addendum B (secondary)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2b_lib import Base2B, excl_mask, fit_eval_lens, fmt  # noqa: E402

LAM = 0.5


def build(ctx):
    lens = Base2B(ctx, LAM, shuf=True)
    print(fmt(f"m2b shuf lam{LAM}", fit_eval_lens(ctx, lens, excl_mask(ctx))), flush=True)
    return lens
