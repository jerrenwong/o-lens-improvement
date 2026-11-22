"""dp1 test: are repeated backward passes on one retained graph identical? argv[1] = 'fla' or 'nofla'."""
import os, sys, json
if sys.argv[1] == "nofla":
    sys.modules["fla"] = None
os.chdir("/root/jpp_lens"); sys.path.insert(0, "/root/jpp_lens/scripts")
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.config import LensConfig
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.utils import source_gradients_summed_over_streams
m = get_hf_model("Qwen/Qwen3.5-2B", attn_implementation="sdpa")
for p in m._hf_model.parameters(): p.requires_grad_(False)
P = json.load(open("/workspace/o-lens-improvement/runs/r1_2b/prompts_raw.json"))[:1]
cfg = LensConfig(hf_model_name="Qwen/Qwen3.5-2B", source_layers=[3, 12, 21], jacobian_rows_per_pass=64,
                 skip_first_n_positions=16, max_seq_len=128, artifacts_base_dir="/root/dp1/tmp", checkpoint_name="x")
tr = LensTrainer(cfg, m, P)
fs = tr._run_fit_forward(P[0])
pos = fs.source_positions_Int_P
cot = t.zeros_like(fs.target_residual_BSF); b = t.arange(64, device=cot.device)
cot[b[:, None], pos[None, :], b[:, None]] = 1.0
outs = []
for i in range(3):
    g = source_gradients_summed_over_streams(fs, cot, retain_graph=True)
    outs.append([x[:, pos].float().mean(1).cpu() for x in g])
cot2 = cot.clone(); cot2[:, pos[::2]] *= -1
g = source_gradients_summed_over_streams(fs, cot2, retain_graph=True)
g = source_gradients_summed_over_streams(fs, cot, retain_graph=False)
outs.append([x[:, pos].float().mean(1).cpu() for x in g])
for j, l in enumerate([3, 12, 21]):
    print(sys.argv[1], "layer", l, [float((outs[i][j] - outs[0][j]).norm() / outs[0][j].norm()) for i in range(4)])
