"""Reserve ~67 GB of GPU memory greedily (pattern of scripts/v3_greedy.py), release it, then run argv[1:] as a
python script in this process (so smaller jobs cannot squeeze in while the 27B model loads)."""
import os, runpy, sys, time
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
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
