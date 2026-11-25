"""run auto/evaluate.py (dev split, --lite) for a list of w12 rows, N in parallel; skips rows with a summary.json.
Usage: python w12_evals.py <model> <jobs.txt> [--par 3]
jobs.txt lines:  <name> <filter> <candidate.py> [ENV=VAL ...]"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

R = "/workspace/o-lens-improvement"
M, jobs = sys.argv[1], sys.argv[2]
par = int(sys.argv[sys.argv.index("--par") + 1]) if "--par" in sys.argv else 3
PY = "/root/jpp_lens/.venv/bin/python"
os.makedirs(f"{R}/runs/w12/evlogs", exist_ok=True)


def run(line):
    p = line.split()
    name, flt, cand, envs = p[0], p[1], p[2], p[3:]
    if os.path.exists(f"{R}/auto/runs/{name}/summary.json"):
        return name, "skip"
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=f"{R}/src")
    env.update(dict(e.split("=", 1) for e in envs))
    t0 = time.time()
    with open(f"{R}/runs/w12/evlogs/{name}.log", "w") as f:
        rc = subprocess.call([PY, f"{R}/auto/evaluate.py", "--lens", f"{R}/{cand}", "--model", M, "--lite",
                              "--filter", flt, "--name", name], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=R)
    return name, f"rc={rc} {time.time() - t0:.0f}s"


lines = [l.strip() for l in open(jobs) if l.strip() and not l.startswith("#")]
with ThreadPoolExecutor(par) as ex:
    for name, st in ex.map(run, lines):
        print(name, st, flush=True)
