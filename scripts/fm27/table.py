"""fm27: dev table from auto/runs/<name>/summary.json (harness aggregate output only)."""
import json
from pathlib import Path

R = Path("/workspace/o-lens-improvement/auto/runs")
ROWS = [("J++", "ask3_jpp_{t}"), ("C", "fm27_diag_{t}"), ("F1 rerank", "fm27_F1_{t}"), ("F2 centred", "fm27_F2_{t}"),
        ("F3 top1-protect", "fm27_F3s_{t}"), ("F4 lam", "fm27_F4_{t}"), ("F5 top1-from-J++10", "fm27_F5s_{t}")]
for t in ("9b", "27b"):
    print(f"\n{t}: name | p10 p1 p10unf p1unf | assoc multihop multiling poetry typo")
    for lab, n in ROWS:
        p = R / n.format(t=t) / "summary.json"
        if not p.exists():
            print(f"{lab:20s} (missing)")
            continue
        d = json.loads(p.read_text())
        tk = " ".join(f"{v:.3f}" for v in d["per_task_pass10"].values())
        print(f"{lab:20s} {d['pass10']:.4f} {d['pass1']:.4f} {d['pass10_unf']:.4f} {d['pass1_unf']:.4f} | {tk}")
