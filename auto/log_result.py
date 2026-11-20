"""Append experiments to results.tsv from auto/runs/<name>/summary.json.  python auto/log_result.py STATUS name="description" ..."""
import json, sys
from pathlib import Path
A = Path(__file__).parent
status = sys.argv[1]
have = {l.split("\t")[0] for l in (A / "results.tsv").read_text().splitlines()[1:]}
with open(A / "results.tsv", "a") as f:
    for arg in sys.argv[2:]:
        name, desc = arg.split("=", 1)
        if name in have:
            print("already logged:", name); continue
        s = json.load(open(A / "runs" / name / "summary.json"))
        g = lambda *ks: next((s[k] for k in ks if k in s), None)
        f.write(f"{name}\t{g('dev_pass10','pass10'):.4f}\t{g('dev_pass1','pass1'):.4f}\t{g('build_seconds','build_s'):.1f}\t{status}\t{desc}\n")
        print("logged", name)
