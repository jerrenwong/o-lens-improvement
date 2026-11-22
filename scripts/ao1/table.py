"""ao1 DEV table (run on the pod: reads /root/ao1/ev/<name>.log for the chosen lam).
    python table.py OUT.md name[=label] ...   (names are auto/runs/<name>)"""
import json
import re
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
out = sys.argv[1]
rows = ["| lens | lam (FIT) | p@10 | p@1 | unf p@10 | assoc | multihop | multiling | poetry | typo |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for a in sys.argv[2:]:
    n, _, lab = a.partition("=")
    f = R / "auto/runs" / n / "summary.json"
    if not f.exists():
        rows.append(f"| {lab or n} | (missing) |")
        continue
    d = json.loads(f.read_text())
    lam = "-"
    lg = Path(f"/root/ao1/ev/{n}.log")
    if lg.exists():
        ch = re.findall(r"\[ao1 (?:mix|prior|perlayer)\S*\] chosen (.*)", lg.read_text())
        ch3 = re.findall(r"\[ao1 three lam(\S+) mu\] chosen (.*)", lg.read_text())
        lam = f"{ch3[-1][0]}, mu {ch3[-1][1]}" if ch3 else (ch[-1] if ch else "-")
    t = d["per_task_pass10"]
    rows.append(f"| {lab or n} | {lam} | {d['pass10']:.3f} | {d['pass1']:.3f} | {d['pass10_unf']:.3f} | " +
                " | ".join(f"{t[k]:.2f}" for k in ("association", "multihop", "multilingual", "poetry", "typo")) + " |")
Path(out).write_text("\n".join(rows) + "\n")
print("\n".join(rows))
