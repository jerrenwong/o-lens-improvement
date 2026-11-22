"""ao1 summary tables (pod; reads /root/ao1/ev logs for FIT choices). python maps_table.py > out.md"""
import json
import re
from pathlib import Path

R = Path("/workspace/o-lens-improvement")


def row(n, lab):
    f = R / "auto/runs" / n / "summary.json"
    if not f.exists():
        return f"| {lab} | (missing: {n}) |"
    d = json.loads(f.read_text())
    t = d["per_task_pass10"]
    lam = "-"
    lg = Path(f"/root/ao1/ev/{n}.log")
    if lg.exists():
        c = re.findall(r"\[ao1 (?:mix|prior|maps mu|three lam\S+ mu|three-step1|perlayer)\] chosen (.+)", lg.read_text())
        lam = ", ".join(c) if c else "-"
    return (f"| {lab} | {lam} | {d['pass10']:.3f} | {d['pass1']:.3f} | {d['pass10_unf']:.3f} | " +
            " | ".join(f"{t[k]:.2f}" for k in ("association", "multihop", "multilingual", "poetry", "typo")) + " |")


H = ("| lens | lam / mu (FIT) | p@10 | p@1 | unf p@10 | assoc | multihop | multiling | poetry | typo |\n"
     "|---|---|---|---|---|---|---|---|---|---|")
TABLES = {
    "9B": [("st1_jppu_9b", "J++ (released, unit)"), ("st1_p3v_t10_A_9b", "P3v topic map alone"),
           ("ao1_9b_t3_maponly", "ridge t3 topic map alone"), ("ao1_9b_s5_maponly", "unstated map alone (ask5 E_9b_s5)"),
           ("ja1_9b_A5_n40_alone", "J_ask A5 n40 alone"), ("ao1_9b_p3v_s5_maps", "NO-JAC: P3v + mu s5"),
           ("ao1_9b_t3_s5_maps", "NO-JAC: t3 + mu s5"), ("ao1_9b_A5n40_p3v_mix", "2-part: A5 n40 + lam P3v"),
           ("ao1_9b_A5n40_p3v_s5_three", "3-part: A5 n40 + lam P3v + mu s5"),
           ("ao1_9b_A5n40_p3v_s6_three", "3-part: A5 n40 + lam P3v + mu s6 (ask6)")],
    "27B": [("st1_jppu_27b", "J++ (released, unit)"), ("st1_p3v_t10_A_27b", "P3v topic map alone"),
            ("ask3_ask3_D_alone_27b", "ridge t3 topic map alone (ask3 D)"),
            ("ao1_27b_s5_maponly", "unstated map alone (ask5 E_27b_s5)"),
            ("ao1_27b_p3v_s5_maps", "NO-JAC: P3v + mu s5"), ("ao1_27b_t3_s5_maps", "NO-JAC: t3 + mu s5")],
}
print("\n## No-Jacobian lens: unit(E_topic h + b) + mu unit(E_unstated h + b'), mu on FIT over {0,.1,.2,.3,.5,1,2,5}")
print("E_unstated = .hf27/ask5/E_<tag>_s5.pt (the s5 map of the 3-part lens). 2B: no unstated-word map exists "
      "(ask5/ask6 maps only for 9B, 27B). 27B: no Jacobian toward the answer exists.")
for k, rs in TABLES.items():
    print(f"\n### {k}\n{H}")
    for r in rs:
        print(row(*r))
