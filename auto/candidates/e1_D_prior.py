"""e1 (D): J++ + LAM * prior-only control (bias = mean target); LAM fixed to the fit-chosen value of e1_A_fitlam (see e1_lib)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_lib import MAIN, build_fixed  # noqa: E402,F401

LAM = 0.5  # fit-chosen in e1_A_fitlam (fit sweep only)


def build(ctx):
    return build_fixed(ctx, MAIN, "prior", LAM)
