"""w1 test (a): --filter f1 reproduces stored dev scores exactly (summary numbers + per-row CSV ranks)."""
import csv as _csv, json
from pathlib import Path
R = Path("/workspace/o-lens-improvement/auto/runs")
pairs = [("w1_repro_jpp_9b", "w1_base_jpp_9b"), ("w1_repro_jpp_2b", "w1_base_jpp_2b"),
         ("w1_repro_alensJPP_9b", "D_al_JPP_9b"), ("w1_repro_alensJPP_2b", "D_al_JPP_2b")]
for t in ("2b", "9b", "27b"):
    pairs += [(f"w1_{t}_{r}_f1", f"D_al_{r}_{t}") for r in ("J0", "JPP0", "none", "J", "JPP")]
for t in ("olmo7b", "gemma4e4b"):
    pairs += [(f"w1_{t}_{r}_f1", f"xm1_{t}_alens_{r}") for r in ("J0", "JPP0", "none", "J", "JPP")]
K = ("pass10", "pass1", "pass10_unf", "pass1_unf", "per_task_pass10", "n_items")
for a, b in pairs:
    if not (R / a / "summary.json").exists() or not (R / b / "summary.json").exists():
        print(f"{a:26s} vs {b:22s} MISSING"); continue
    sa, sb = (json.loads((R / x / "summary.json").read_text()) for x in (a, b))
    same = all(sa[k] == sb[k] for k in K)
    rd = lambda p: [{k: v for k, v in r.items() if k != "lens"} for r in _csv.DictReader(open(p))]
    csv = all(rd(R / a / f) == rd(R / b / f) for f in ("readout_filtered.csv", "readout_unfiltered.csv") if (R / b / f).exists())
    print(f"{a:26s} vs {b:22s} summary {'IDENTICAL' if same else 'DIFF'}  csv-ranks {'IDENTICAL' if csv else 'DIFF'}  "
          f"p10 {sa['pass10']:.4f}/{sb['pass10']:.4f} p1 {sa['pass1']:.4f}/{sb['pass1']:.4f} filter={sa.get('filter')}")
