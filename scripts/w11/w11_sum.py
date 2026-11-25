"""w11: one line per run (dev top-10 / top-1, unfiltered, chosen params from the log). python w11_sum.py <prefix>"""
import json, re, sys
from pathlib import Path
R = Path("/workspace/o-lens-improvement")
for d in sorted((R / "auto/runs").glob(sys.argv[1] + "*")):
    s = d / "summary.json"
    if not s.exists():
        continue
    s = json.loads(s.read_text())
    lg = R / f"runs/w11/logs/{d.name}.log"
    ch = " ".join(re.findall(r"chosen (?:norm \S+ nmin \d+|beta \S+|lam \S+)", lg.read_text())) if lg.exists() else ""
    print(f"{d.name:32s} {s['pass10']:.3f} / {s['pass1']:.3f} | unf {s['pass10_unf']:.3f} / {s['pass1_unf']:.3f} | {ch}")
