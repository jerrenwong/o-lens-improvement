"""ao1 variant (b): the authors' expert-weight search (jpp_cli fit-weights, imported from /root/jpp_lens, not edited)
run on the COMBINED logits: the K=8 J_ask experts plus one extra "expert" whose transport is c_l * unit(E h + b)
(the regression ask-map term; c_l = median rms of the expert transports on the fitting items, only for the
search's conditioning: the weight is free and signed). Same items (fit:0), same search config, same starts.

    python joint_fit.py --experts X_experts_K8.pt --map E.pt --out DIR/name
writes  name_jpp.pt (stock lens from the 8 expert weights), name.pt (plain dict), name_mapw.json {layer: w_9 * c_l}
Lens (auto/candidates/ao1_joint.py): v = unit(h @ J_l^T + m_l unit(E h + b)), J_l = sum_e w_e J_e, m_l = w_9 c_l."""
import argparse
import json
import os
import sys

sys.modules["fla"] = None
import torch as t  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--experts", required=True)
a.add_argument("--map", required=True)
a.add_argument("--out", required=True)
a.add_argument("--model", default="Qwen/Qwen3.5-9B")
a.add_argument("--layers", default="4,8,12,16,20,24,28")
a.add_argument("--cache", default="/root/r2_readout_resid_cache_9b.pt")
args = a.parse_args()

os.chdir("/root/jpp_lens")
sys.path.insert(0, "/root/jpp_lens/scripts")
import jpp_cli  # noqa: E402
from workspace_lens.fitting.condense_experts import fitter as F  # noqa: E402
from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians  # noqa: E402

M = t.load(args.map, map_location="cpu")
MAPS = {int(l): (v["mx"].float(), v["W"].float() * v["scale"], v["my"].float()) for l, v in M.items()}
C = {}


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


orig_T = F.ExpertWeightingFitter._expert_transports_PEF


def transports(self, layer, idx):
    T = orig_T(self, layer, idx)  # [P, E, F]
    h = self.residuals.residuals_L_dict_PN[layer].to(device=self.device, dtype=t.float32)[idx]
    mx, W, my = (x.to(self.device) for x in MAPS[layer])
    m = unit((h - mx) @ W + my)
    c = float(T.pow(2).mean(-1).sqrt().median())
    C[layer] = c
    return t.cat([T, (c * m)[:, None]], 1)


F.ExpertWeightingFitter._expert_transports_PEF = transports
orig_comb = ExpertJacobians.combine
W9 = {}


def combine(self, w, *, checkpoint_name):
    for l, v in w.items():
        W9[int(l)] = v.tolist()
    return orig_comb(self, {l: v[:-1] for l, v in w.items()}, checkpoint_name=checkpoint_name)


ExpertJacobians.combine = combine
csv = "/workspace/o-lens-improvement/data/model_correctness.csv"
rc = jpp_cli.main(["fit-weights", "--hf-model-name", args.model, "--layers", args.layers,
                   "--experts", args.experts, "--checkpoint-name", "ao1_joint", "--items", "fit:0",
                   "--correctness-csv", csv, "--cache-path", args.cache, "--out", args.out + "_jpp.pt"])
assert W9 and len(next(iter(W9.values()))) == 9, W9
json.dump({str(l): W9[l][-1] * C[l] for l in W9}, open(args.out + "_mapw.json", "w"))
json.dump(dict(weights9=W9, c=C), open(args.out + "_info.json", "w"), indent=1)
from workspace_lens.utils import load_lens_file  # noqa: E402
L = load_lens_file(args.out + "_jpp.pt", hf_model_name=args.model)
t.save({int(l): v.float().cpu() for l, v in L.jacobians_L_dict_FN.items()}, args.out + ".pt")
print("[ao1 joint] map weights", json.load(open(args.out + "_mapw.json")), "c", C, flush=True)
