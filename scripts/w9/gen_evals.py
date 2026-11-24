"""write an eval spec file: python3 gen_evals.py <out> <tag> <n:28k|98k> [archs] [losses] [rows] [filters] [suffix]"""
import sys
R = "/workspace/o-lens-improvement"
out, tag, n = sys.argv[1:4]
archs = (sys.argv[4] if len(sys.argv) > 4 else "lin,mlp512,mlp2048").split(",")
losses = (sys.argv[5] if len(sys.argv) > 5 else "l2,ce").split(",")
rows = (sys.argv[6] if len(sys.argv) > 6 else "none,JPP").split(",")
flts = (sys.argv[7] if len(sys.argv) > 7 else "f4,f1").split(",")
suf = sys.argv[8] if len(sys.argv) > 8 else ""
M = {"9b": "Qwen/Qwen3.5-9B", "2b": "Qwen/Qwen3.5-2B"}[tag]
with open(out, "a") as f:
    for flt in flts:
        for row in rows:
            for a in archs:
                for lo in losses:
                    s = f"{a}_{lo}_{n}{suf}"
                    E = f"{R}/.hf27/w9/maps/E_{tag}_topic_{s}.pt"; E2 = f"{R}/.hf27/w9/maps/E_{tag}_spec_{s}.pt"
                    f.write(f"w9_{tag}_{s}_{row}_{flt}|ALENS_BASE={row} ALENS_E={E} ALENS_E2={E2}|{M}|{flt}\n")
