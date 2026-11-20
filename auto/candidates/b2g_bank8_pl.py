"""Idea 8: own expert bank (BANK8), J++-style signed weights, shared=False; ridge lam toward J++ chosen by
5-fold x3 CV on the fit split (see b2g_common)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2g_common import BANK8, bank_lens  # noqa: E402


def build(ctx):
    return bank_lens(ctx, BANK8, shared=False, cv_table=True)
