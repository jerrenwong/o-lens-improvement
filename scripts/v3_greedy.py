"""Run the three 27B v3 evaluations in ONE process that first reserves GPU memory greedily (grabbing blocks as other
jobs release them), so smaller jobs cannot squeeze in while the 27B model loads."""
import gc, os, runpy, sys, time
import torch
os.chdir("/workspace/o-lens-improvement")
need = 67 * 2**30; blocks = []; got = 0; t0 = time.time()
while got < need:
    for sz in (8, 4, 2, 1):
        try:
            blocks.append(torch.empty(sz * 2**30, dtype=torch.uint8, device="cuda")); got += sz * 2**30; break
        except torch.OutOfMemoryError:
            continue
    else:
        time.sleep(5)
print(f"reserved {got / 2**30:.0f} GB after {time.time() - t0:.0f}s", flush=True)
del blocks
for c in ("jpp", "m27_A", "m27_C_shuf"):
    sys.argv = ["auto/evaluate_v3.py", "--lens", f"auto/candidates/{c}.py", "--name", f"V3_{c}_27b", "--model", "Qwen/Qwen3.6-27B", "--split", "test", "--orchestrator"]
    print(f"=== {c} start {time.strftime('%H:%M:%S')}", flush=True)
    try:
        ns = runpy.run_path("auto/evaluate_v3.py", run_name="__main__")
    except SystemExit as e:
        print("exit", e.code, flush=True)
    except Exception as e:
        print("FAILED", c, repr(e)[:300], flush=True)
    ns = None; gc.collect()
    print(f"=== {c} end {time.strftime('%H:%M:%S')}", flush=True)
print("ALL DONE", flush=True)
