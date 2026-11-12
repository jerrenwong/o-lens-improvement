"""Run the jpp_lens CLI (evaluate / probe-swap / ...) with the ``trained`` lens type registered.

    python -m olens.evaluate evaluate --hf-model-name Qwen/Qwen3.5-2B --layers 3,6,9,12,15,18,21 \
        --lens /abs/path/a.pt logit ... --correctness-csv /abs/.../model_correctness.csv --out ...

Also: ``python -m olens.evaluate grade-correctness --hf-model-name M --out CSV`` adds a column
for model M to a copy of jpp_lens's model-correctness CSV (greedy next token == target).
jpp_cli reads its eval data by relative path, so this runs from the jpp_lens checkout:
give absolute paths for everything else.
"""

from __future__ import annotations

import os
import sys

import olens  # noqa: F401  registers TrainedLens with BaseLens

JPP = os.environ.get("JPP_LENS_DIR", "/root/jpp_lens")


def grade_correctness(argv: list[str]) -> None:
    import argparse

    import pandas as pd
    from lens_evals.readout_evals.readout_eval_items import (
        MACRO_EVALS,
        MODEL_CORRECTNESS_CSV,
        READOUT_EVALS_DATA_DIR,
        load_readout_eval_items,
    )
    from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
    from workspace_lens import get_hf_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    items = load_readout_eval_items(READOUT_EVALS_DATA_DIR, slugs=MACRO_EVALS)
    runner = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=512)
    graded = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)
    base = pd.read_csv(args.out if os.path.exists(args.out) else MODEL_CORRECTNESS_CSV)
    base = base.drop(columns=[args.hf_model_name], errors="ignore")
    merged = base.merge(graded, on=["eval", "item"], how="outer")
    merged.to_csv(args.out, index=False)
    col = graded[args.hf_model_name]
    print(graded.assign(c=col.astype(str)).groupby(["eval", "c"]).size().unstack(fill_value=0))


def main() -> None:
    os.chdir(JPP)
    if len(sys.argv) > 1 and sys.argv[1] == "grade-correctness":
        grade_correctness(sys.argv[2:])
        return
    sys.path.insert(0, os.path.join(JPP, "scripts"))
    import jpp_cli

    raise SystemExit(jpp_cli.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
