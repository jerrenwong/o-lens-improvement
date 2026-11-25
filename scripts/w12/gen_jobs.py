"""write an eval job list for w12_evals.py. python3 gen_jobs.py <tag> <out.txt> cfg:base[,base][:al] ...
  base in none|J|JPP; ':al' adds the A-lens (CE 98k) on top (W12_ALENS=1); rows already scored are skipped by w12_evals."""
import sys
R = "/workspace/o-lens-improvement"
tag, out = sys.argv[1], sys.argv[2]
D = f"{R}/.hf27/w12/{tag}/maps"
L = []
for spec in sys.argv[3:]:
    p = spec.split(":")
    cfg, bases, al = p[0], p[1].split(","), len(p) > 2 and p[2] == "al"
    for b in bases:
        for f in ("f4", "f1"):
            name = f"w12_{tag}_{cfg}_{b}{'_al' if al else ''}_{f}"
            L.append(f"{name} {f} auto/candidates/w12_lens.py W12_MAP={D}/{cfg}.pt W12_BASE={b}" + (" W12_ALENS=1" if al else ""))
open(out, "w").write("\n".join(L) + "\n")
print(len(L), "jobs")
