"""w3: run scripts/ao1/ao1_fit.py unchanged, with a per-process GPU memory cap (env W3_MEM_GB, default 45) so the
caching allocator frees and retries instead of growing past the cap (the ao1 9B refit grew to 71 GB and OOM'd).
    python scripts/w3/fitwrap.py <ao1_fit.py args...>
"""
import os
import runpy
import sys

if os.environ.get("JA1_FLA") != "1":
    sys.modules["fla"] = None  # same as ao1_fit.py; must precede any transformers import
import torch  # noqa: E402

gb = float(os.environ.get("W3_MEM_GB", "45"))
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / tot))
print(f"[w3] memory cap {gb:.0f} GiB ({gb * 2**30 / tot:.3f} of device)", flush=True)
# Leak fix: ao1_fit's prefix cache binds update_recurrent_state = MethodType(f, layer_cache) -> a reference cycle,
# so each prompt's cache (B replicas, graph-attached states) is freed only by the cyclic gc, which GPU memory does not
# trigger (memory grew ~1 GB/prompt here; the ao1 9B refit reached 71 GB). Collect after every prompt; no math change.
import gc  # noqa: E402

sys.path.insert(0, "/root/jpp_lens/scripts")
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer as _T  # noqa: E402

_fs = _T.fit_step


def _fit_step(self, *a, **k):
    out = _fs(self, *a, **k)
    gc.collect()
    torch.cuda.empty_cache()
    return out


_T.fit_step = _fit_step
F = "/workspace/o-lens-improvement/scripts/ao1/ao1_fit.py"
sys.argv = [F] + sys.argv[1:]
runpy.run_path(F, run_name="__main__")
