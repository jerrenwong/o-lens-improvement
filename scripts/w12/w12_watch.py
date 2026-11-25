"""w12 watcher (pod): as soon as all 7 layer files of a config exist, merge them (<cfg>.pt) and run its dev evals
(alone, filters f4 and f1; extra bases via --bases), up to --par evaluate.py processes at once. Stops when every config
is evaluated. Usage: python w12_watch.py <model> <cfg1,cfg2,...> [--bases none] [--par 3]"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

R = "/workspace/o-lens-improvement"
M = sys.argv[1]
cfgs = sys.argv[2].split(",")
bases = sys.argv[sys.argv.index("--bases") + 1].split(",") if "--bases" in sys.argv else ["none"]
par = int(sys.argv[sys.argv.index("--par") + 1]) if "--par" in sys.argv else 3
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
D = f"{R}/.hf27/w12/{TAG}/maps"
PY = "/root/jpp_lens/.venv/bin/python"
os.makedirs(f"{R}/runs/w12/evlogs", exist_ok=True)


def ev(name, flt, envs):
    if os.path.exists(f"{R}/auto/runs/{name}/summary.json"):
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=f"{R}/src", **envs)
    t0 = time.time()
    with open(f"{R}/runs/w12/evlogs/{name}.log", "w") as f:
        rc = subprocess.call([PY, f"{R}/auto/evaluate.py", "--lens", f"{R}/auto/candidates/w12_lens.py", "--model", M,
                              "--lite", "--filter", flt, "--name", name], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=R)
    print(f"{name} rc={rc} {time.time() - t0:.0f}s", flush=True)


import torch  # noqa: E402

pending = list(cfgs)
futs = []
with ThreadPoolExecutor(par) as ex:
    while pending:
        for c in list(pending):
            fs = [f"{D}/{c}_L{l}.pt" for l in LAYERS]
            if all(os.path.exists(f) for f in fs):
                time.sleep(5)
                torch.save({l: torch.load(f) for l, f in zip(LAYERS, fs)}, f"{D}/{c}.pt")
                print(f"merged {c}", flush=True)
                pending.remove(c)
                for b in bases:
                    for flt in ("f4", "f1"):
                        futs.append(ex.submit(ev, f"w12_{TAG}_{c}_{b}_{flt}", flt, dict(W12_MAP=f"{D}/{c}.pt", W12_BASE=b)))
        time.sleep(30)
    for f in futs:
        f.result()
print("watch done", flush=True)
