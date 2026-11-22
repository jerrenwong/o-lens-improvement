"""xm1: DEV table from auto/runs/xm1_<tag>_<lens>/summary.json.   python scripts/xm1/table.py <tag>"""
import json
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
tag = sys.argv[1]
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
print(f"| lens | pass@10 | pass@1 | unf pass@10 | " + " | ".join(TASKS) + " | n |")
print("|---|---|---|---|" + "---|" * len(TASKS) + "---|")
for c in ["jlens", "jppraw", "jppu", "jppu_full", "L", "Lshuf", "Lprior", "alone", "alens_J0", "alens_JPP0", "alens_none", "alens_J", "alens_JPP"]:
    p = R / f"auto/runs/xm1_{tag}_{c}/summary.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    assert s["split"] == "dev"
    print(f"| {c} | {100*s['pass10']:.1f} | {100*s['pass1']:.1f} | {100*s['pass10_unf']:.1f} | "
          + " | ".join(f"{100*s['per_task_pass10'].get(t, float('nan')):.1f}" for t in TASKS) + f" | {s['n_items']} |")
