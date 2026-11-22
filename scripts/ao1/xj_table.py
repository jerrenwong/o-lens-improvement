"""ao1 XJ table: FIT (from runs/ao1/<tag>/xj/<name>.json) next to DEV (auto/runs/ao1_<tag>_xj_<name>).
    python xj_table.py 9b n40 [extra baseline run names...]"""
import json
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
tag, k = sys.argv[1], sys.argv[2]
D = R / "runs/ao1" / tag / "xj"
DESC = {"J0": "J++ refit experts (8) alone, same search", "J3_0": "released J++ alone (1 weight)",
        "J1": "8 J++ exp + A2 pooled ask", "J1c": "ctl: 8 J++ exp + A2 layer-permuted",
        "J1w": "ctl: 8 J++ exp + wrong-passage A4", "J2": "8 J++ exp + 8 ask exp",
        "J2c": "ctl: 8 J++ exp + 8 ask exp layer-permuted", "J3a": "released J++ + A2",
        "J3ac": "ctl: released J++ + A2 permuted", "J3b": "released J++ + 8 ask exp",
        "J3bc": "ctl: released J++ + 8 ask exp permuted", "J4": "J2 + ask-map E (affine, b kept)",
        "J4c": "ctl: J2c + ask-map E"}
rows = ["| name | basis | K | FIT p@10 | FIT p@1 | DEV p@10 | DEV p@1 | DEV unf | assoc | multihop | multiling | poetry | typo |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
names = ["J0", "J3_0"] + [f"{b}_{k}" for b in ("J1", "J1c", "J1w", "J2", "J2c", "J3a", "J3ac", "J3b", "J3bc", "J4", "J4c")]
for n in names:
    f = D / f"{n}.json"
    if not f.exists():
        continue
    fi = json.loads(f.read_text())
    K = len(next(iter(fi["layers"].values()))["w"])
    s = R / "auto/runs" / f"ao1_{tag}_xj_{n}" / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        t = d["per_task_pass10"]
        dev = f"{d['pass10']:.3f} | {d['pass1']:.3f} | {d['pass10_unf']:.3f} | " + " | ".join(
            f"{t[x]:.2f}" for x in ("association", "multihop", "multilingual", "poetry", "typo"))
    else:
        dev = "- | - | - | | | | |"
    rows.append(f"| {n} | {DESC[n.split('_')[0] if n != 'J3_0' else n]} | {K} | {fi['fit_p10']:.3f} | {fi['fit_p1']:.3f} | {dev} |")
print("\n".join(rows))
