"""w9: forward-pass timing check (collection throughput diagnosis). python w9_speedtest.py <model> <threads>"""
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

torch.set_num_threads(int(sys.argv[2]))
torch.set_grad_enabled(False)
t0 = time.time()
m = get_hf_model(sys.argv[1], attn_implementation="sdpa")
print("load", time.time() - t0, next(m._hf_model.parameters()).dtype, flush=True)
for bs, L in ((128, 200), (128, 201), (128, 202), (128, 150), (100, 150), (128, 64), (128, 65)):
    ids = torch.randint(1000, 50000, (bs, L), device="cuda")
    for r in range(2):
        torch.cuda.synchronize(); t = time.time(); m.forward(ids); torch.cuda.synchronize()
        print(f"bs {bs} L {L} rep {r}: {time.time() - t:.2f}s", flush=True)
