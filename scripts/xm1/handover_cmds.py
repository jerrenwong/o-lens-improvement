"""xm1: print the orchestrator's held-out commands for a model (test split + v4), one line per lens.
    python scripts/xm1/handover_cmds.py <model> <tag>"""
import sys

M, T = sys.argv[1], sys.argv[2]
R = "/workspace/o-lens-improvement"
PRE = f"cd {R} && HF_HOME={R}/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH={R}/src /root/jpp_lens/.venv/bin/python"
prim = ["jppu", "L", "Lshuf", "Lprior"]
print("### Primary (PREREG_v5): test split")
for c in prim:
    print(f"{PRE} auto/evaluate_x.py --lens auto/candidates/xm1_{c}.py --model {M} --split test --orchestrator --lite --name xm1_{T}_{c}_test")
print("\n### Primary (PREREG_v5): v4")
for c in prim:
    print(f"{PRE} auto/evaluate_x_v4.py --lens auto/candidates/xm1_{c}.py --model {M} --split test --orchestrator --lite --name xm1_{T}_{c}_v4")
env = f"ALENS_E={R}/.hf27/xm1/E_{T}_t3.pt ALENS_E2={R}/.hf27/xm1/E_{T}_s5.pt"
print("\n### Amendment 1 (A-lens rows; ALENS_E/ALENS_E2 MUST be set: alens.py's default path picks maps by d_model and would load another model's maps)")
for split, sc in (("test", "evaluate_x.py"), ("v4", "evaluate_x_v4.py")):
    for b in ["J0", "JPP0", "none", "J", "JPP"]:
        print(f"cd {R} && ALENS_BASE={b} {env} HF_HOME={R}/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH={R}/src /root/jpp_lens/.venv/bin/python "
              f"auto/{sc} --lens auto/candidates/alens.py --model {M} --split test --orchestrator --lite --name xm1_{T}_alens_{b}_{split}")
p = f"xm1_{T}"
print("\n### Bootstrap (5000 draws within task), per held-out set S in {test, v4}")
print(f"cd {R} && python scripts/r2_boot.py {p}_L_S-{p}_jppu_S {p}_L_S-{p}_Lshuf_S {p}_L_S-{p}_Lprior_S   # replace S")
