"""ex2: score many spec lenses on DEV through the unmodified auto/evaluate.py main(), in one process.
Only the loaders are memoised (Bench.load, the --lite model, load_lens_file), so the model/bench are read once;
every lens goes through evaluate.main() exactly as `python auto/evaluate.py --lens auto/candidates/ex2_spec.py
--model M --name ex2_<name> --lite` would. Skips names whose auto/runs/ex2_<name>/summary.json exists.
On a crash (e.g. CUDA OOM from other jobs) waits and retries.
    python scripts/ex2/multi_eval.py --model 9b --specs runs/ex2/specs_9b.json [--names a,b,...]
"""
import argparse
import functools
import importlib.util
import json
import os
import sys
import time

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["9b", "2b"], required=True)
ap.add_argument("--specs", required=True)
ap.add_argument("--names", default=None)
ap.add_argument("--retries", type=int, default=20)
args = ap.parse_args()
MODEL = {"9b": "Qwen/Qwen3.5-9B", "2b": "Qwen/Qwen3.5-2B"}[args.model]

sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/perf1")
import torch  # noqa: E402
import lite_model  # noqa: E402
import workspace_lens.utils as wu  # noqa: E402
from olens.readout import Bench  # noqa: E402

_glm = lite_model.get_lite_model
lite_model.get_lite_model = functools.lru_cache(maxsize=None)(_glm)
_bl = Bench.load.__func__
_bench_cache = {}


def _bench_load(cls, path):
    k = str(path)
    if k not in _bench_cache:
        _bench_cache[k] = _bl(cls, path)
    return _bench_cache[k]


Bench.load = classmethod(_bench_load)
_llf = wu.load_lens_file
wu.load_lens_file = functools.lru_cache(maxsize=None)(_llf)

spec = importlib.util.spec_from_file_location("ex2_evaluate", f"{R}/auto/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

specs = json.load(open(args.specs))
names = args.names.split(",") if args.names else list(specs)
os.environ["EX2_SPECS"] = os.path.abspath(args.specs)
for nm in names:
    run = f"ex2_{nm}"
    summ = f"{R}/auto/runs/{run}/summary.json"
    if os.path.exists(summ):
        print(f"[multi] skip {run} (done)", flush=True)
        continue
    os.environ["EX2_NAME"] = nm
    for attempt in range(args.retries):
        sys.argv = ["auto/evaluate.py", "--lens", f"{R}/auto/candidates/ex2_spec.py", "--model", MODEL,
                    "--name", run, "--lite"]
        t0 = time.time()
        try:
            ev.main()
        except SystemExit as e:
            print(f"[multi] {run} exited {e.code} (attempt {attempt})", flush=True)
        except Exception as e:  # noqa: BLE001  (e.g. OOM while scoring)
            print(f"[multi] {run} raised {e!r} (attempt {attempt})", flush=True)
        torch.cuda.empty_cache()
        if os.path.exists(summ):
            print(f"[multi] done {run} in {time.time() - t0:.0f}s", flush=True)
            break
        time.sleep(60)
    else:
        print(f"[multi] FAILED {run}", flush=True)
print("[multi] all done", flush=True)
