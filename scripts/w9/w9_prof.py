"""w9: profile the first forward call at a new input shape (collection overhead diagnosis)."""
import cProfile
import pstats
import sys

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

torch.set_num_threads(8)
torch.set_grad_enabled(False)
m = get_hf_model(sys.argv[1], attn_implementation="sdpa")
m.forward(torch.randint(1000, 50000, (8, 40), device="cuda"))
ids = torch.randint(1000, 50000, (16, 77), device="cuda")
pr = cProfile.Profile(); pr.enable(); m.forward(ids); torch.cuda.synchronize(); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
