"""Concatenate the offline dev-split readout CSVs (auto/runs/<name>/readout_*.csv) of the r1 lenses into
runs/r1_2b/eval/readout_{filtered,unfiltered}.csv for scripts/analyze.py (paired bootstrap).
    python scripts/r1_collect.py r1_A r1_B r1_C r1_D r1_prefit_jpp r1_prefit_jlens"""
import sys
from pathlib import Path
import pandas as pd
R = Path("/workspace/o-lens-improvement"); out = R / "runs/r1_2b/eval"; out.mkdir(parents=True, exist_ok=True)
for kind in ("filtered", "unfiltered"):
    df = pd.concat([pd.read_csv(R / "auto/runs" / n / f"readout_{kind}.csv") for n in sys.argv[1:]])
    print(kind, df.groupby("lens")["item"].nunique().to_dict())
    df.to_csv(out / f"readout_{kind}.csv", index=False)
