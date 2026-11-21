"""lr1: results table from auto/runs/lr1_*/summary.json and the eval logs (FIT pass@10 line)."""
import json
import re
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
names = sorted(p.name for p in (R / "auto/runs").glob("lr1_*") if (p / "summary.json").exists())
for extra in ("r2_A_full", "r2_C_full", "r2_released"):
    names.append(extra)
print(f"{'run':18s} {'FIT10':>6s} {'FIT1':>6s} {'dev10':>6s} {'dev1':>6s} {'unf10':>6s}  " + " ".join(t[:5] for t in TASKS) + "   wfit(mean layer)")
for n in names:
    s = json.loads((R / "auto/runs" / n / "summary.json").read_text())
    log = R / ".hf27/lr1/logs" / f"ev_{n}.log"
    f10 = f1 = float("nan")
    if log.exists():
        txt = log.read_text()
        m = re.search(r"FIT_pass10 ([\d.]+)", txt); f10 = float(m.group(1)) if m else f10
        m = re.search(r"FIT_pass1 ([\d.]+)", txt); f1 = float(m.group(1)) if m else f1
    wf = ""
    mm = re.match(r"lr1_([AC])_(V2|V3b)_r(\d+)", n)
    if mm:
        wj = R / f".hf27/lr1/{mm.group(1)}/{mm.group(1)}_{mm.group(2)}_r{mm.group(3)}_expert_weights.json"
        if wj.exists():
            d = json.loads(wj.read_text())["by_layer"]
            wf = f"{sum(v['train_macro_pass'] for v in d.values()) / len(d):.3f}"
    pt = s["per_task_pass10"]
    print(f"{n:18s} {f10:6.3f} {f1:6.3f} {s['pass10']:6.4f} {s['pass1']:6.4f} {s['pass10_unf']:6.4f}  "
          + " ".join(f"{pt.get(t, float('nan')):5.2f}" for t in TASKS) + f"   {wf}")
