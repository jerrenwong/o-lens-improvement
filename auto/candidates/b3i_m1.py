"""b3i: measurement run. Returns plain J++; build() prints the label-free ceiling M1 (J++ on WikiText period
activations vs the model's own elicited topic word), held-out agreement of the elicit-distilled ridge map E_l
per layer (token ceiling, with mean-answer and shuffled-target controls), and fit-split scores of the variants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3i_lib import build_maps, fit_report, ELens  # noqa: E402


def build(ctx):
    maps, info = build_maps(ctx, diagnostics=True, shuffle_control=True)
    fit_report(ctx, maps, info)
    return ELens(ctx.JPP, maps, "jpp")
