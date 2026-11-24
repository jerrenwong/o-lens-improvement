"""w1: dev table (top-10 / top-1 filtered, chosen lambda, masked ids) -> runs/w1/dev_table.md"""
import json, re
from pathlib import Path
R = Path("/workspace/o-lens-improvement")
ROWS = [("J0", "J-lens"), ("JPP0", "J++"), ("none", "A-lens alone"), ("J", "J-lens + A-lens"), ("JPP", "J++ + A-lens")]
TAGS = ["2b", "9b", "27b", "olmo7b", "gemma4e4b"]
out = ["# W1 dev table (dev split, --lite, filtered score at each level; lambda re-chosen on fit under the level)", ""]
for t in TAGS:
    body, masks = [], {}
    for code, label in ROWS:
        cells = []
        for lv in ("f1", "f2", "f3"):
            n = f"w1_{t}_{code}_{lv}"; sp = R / "auto/runs" / n / "summary.json"; lp = R / "runs/w1/filt_logs" / f"{n}.log"
            if not sp.exists():
                cells.append("–"); continue
            s = json.loads(sp.read_text()); log = lp.read_text() if lp.exists() else ""
            m = re.search(r"chosen lam ([0-9.]+)", log); lam = m.group(1) if m else ("0" if code.endswith("0") else "–")
            mm = re.search(r"\[w1\] filter (f\d): (\d+) ids masked, (no groups|\d+ groups)", log)
            if mm: masks[lv] = mm.group(2) + ("" if mm.group(3) == "no groups" else f" ({mm.group(3)})")
            cells.append(f"{s['pass10']:.3f} / {s['pass1']:.3f} (λ {lam})")
        body.append(f"| {label} | " + " | ".join(cells) + " |")
    if all("–" == c for row in body for c in row.split(" | ")[1:-1]):
        continue
    unf = ""
    out += [f"## {t}", "", "masked ids: " + ", ".join(f"{k} {v}" for k, v in sorted(masks.items())), "",
            "| row | f1 top-10 / top-1 | f2 | f3 |", "|---|---|---|---|"] + body + [""]
(R / "runs/w1/dev_table.md").write_text("\n".join(out))
print("\n".join(out))
