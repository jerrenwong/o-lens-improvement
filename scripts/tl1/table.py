"""tl1 DEV table: J, J++, A alone, T alone, J + A, J + T, J++ + A, J++ + T (pass@10 / pass@1 filtered, %), + T-map fit."""
import json
from pathlib import Path
R = Path("/workspace/o-lens-improvement")
for t in ("9b", "2b", "olmo7b", "gemma4e4b", "27b"):
    ref = (lambda r: f"D_al_{r}_{t}") if t in ("9b", "2b", "27b") else (lambda r: f"xm1_{t}_alens_{r}")
    cols = [("J", ref("J0")), ("J++", ref("JPP0")), ("A", ref("none")), ("T", f"D_tl_none_{t}"), ("J+A", ref("J")),
            ("J+T", f"D_tl_J_{t}"), ("J+++A", ref("JPP")), ("J+++T", f"D_tl_JPP_{t}")]
    out = []
    for lab, n in cols:
        p = R / "auto/runs" / n / "summary.json"
        if p.exists():
            s = json.loads(p.read_text()); out.append(f"{lab} {100*s['pass10']:.1f}/{100*s['pass1']:.1f}")
        else:
            out.append(f"{lab} -")
    print(t, " | ".join(out))
