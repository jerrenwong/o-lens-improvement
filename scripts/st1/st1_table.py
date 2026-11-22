"""st1 dev table: python st1_table.py <tag> (reads auto/runs/st1_*_<tag>/summary.json and /root/st1/eval_*.log)."""
import json
import re
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement")
tag = sys.argv[1]
V = ["B0", "p1_t10", "p2_t05", "p2_t10", "p2_t20", "p3_t05", "p3_t10", "p3_t20", "p1_t00", "p1f_t10", "p2f_t10", "p3v_t10"]


def load(n):
    p = R / "auto/runs" / n / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def logi(n):
    p = Path(f"/root/st1/eval_{n}.log")
    if not p.exists():
        return None, None
    s = p.read_text()
    lam = re.findall(r"chosen lam ([\d.]+)", s)
    lam = float(lam[-1]) if lam else None
    fit = None
    if lam is not None:
        m = re.findall(rf"lam{lam}\s+p10 ([\d.]+) p1 ([\d.]+)", s)
        fit = float(m[-1][0]) + float(m[-1][1]) if m else None
    return lam, fit


rows = [("J++raw", load(f"st1_jpp_{tag}"), (None, None), None), ("J++unit", load(f"st1_jppu_{tag}"), (0.0, None), None),
        ("Ju x1.37", load(f"st1_jppu_s137_{tag}"), (0.0, None), None), ("Ju x0.73", load(f"st1_jppu_s073_{tag}"), (0.0, None), None),
        ("Ju rep", load(f"st1_jppu_rep_{tag}"), (0.0, None), None), ("B0 rep", load(f"st1_B0_rep_{tag}"), logi(f"st1_B0_rep_{tag}"), None)]
for v in V:
    rows.append((v, load(f"st1_{v}_{tag}"), logi(f"st1_{v}_{tag}"), load(f"st1_{v}_A_{tag}")))
tasks = None
print(f"== {tag} DEV (lite). L = unit(J++ h) + lam unit(E h + b); A = map alone")
for name, s, (lam, fit), a in rows:
    if s is None:
        print(f"{name:8s} missing"); continue
    tasks = tasks or list(s["per_task_pass10"])
    pt = " ".join(f"{s['per_task_pass10'][t]:.3f}" for t in tasks)
    al = f" | A p10 {a['pass10']:.3f} p1 {a['pass1']:.3f} unf {a['pass10_unf']:.3f}" if a else ""
    print(f"{name:8s} lam {lam if lam is not None else '-':<5} FIT p1+p10 {fit if fit else float('nan'):.4f} | "
          f"p10 {s['pass10']:.3f} p1 {s['pass1']:.3f} unf10 {s['pass10_unf']:.3f} | {pt}{al}")
print("tasks:", tasks)

print("\n== bf16 rounding noise: same lens, dev output scaled by s in {1, .61, .83, 1.19, 1.37} (lam fixed by FIT)")
import statistics as st  # noqa: E402
for base, n1 in (("jppu", f"st1_jppu_{tag}"), ("B0", f"st1_B0_{tag}"), ("p3_t05", f"st1_p3_t05_{tag}"), ("p3_t10", f"st1_p3_t10_{tag}")):
    names = [n1] + [f"st1_n_{base}_s{s}_{tag}" for s in ("061", "083", "119")] + \
            [f"st1_jppu_s137_{tag}" if base == "jppu" else f"st1_n_{base}_s137_{tag}"]
    ss = [x for x in (load(n) for n in names) if x]
    if len(ss) < 2:
        print(f"{base:8s} n={len(ss)}"); continue
    out = []
    for k in ("pass10", "pass1", "pass10_unf"):
        v = [x[k] for x in ss]
        out.append(f"{k} mean {st.mean(v):.3f} sd {st.stdev(v):.3f} range {min(v):.3f}-{max(v):.3f}")
    print(f"{base:8s} n={len(ss)}: " + " | ".join(out))
