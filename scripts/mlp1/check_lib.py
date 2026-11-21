"""check auto/candidates/mlp1_lib.apply_f == mlp1_train.Net forward for every exported group at layer 20 (CPU)."""
import os, sys, json
os.environ["MLP1_DEV"] = "cpu"
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/src")
import workspace_lens, olens  # noqa  (as evaluate.py: imported before the candidates dir is on sys.path)
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
sys.path.insert(0, "/workspace/o-lens-improvement/auto/candidates")
from mlp1_common import *  # noqa
from mlp1_train import DEFAULT, Net, run_name
import mlp1_lib
torch.set_grad_enabled(False)
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
X = torch.load(f"{BIG}/X_L20.pt")[:300].float()
final = json.load(open(f"{R}/scripts/mlp1/final_all.json"))
m = torch.load(f"{OUT}/ridge_L20.pt")
for g in list(final) + ["ridge"]:
    F = mlp1_lib.load_f(g, "cpu")
    a = mlp1_lib.apply_f(F[20], X)
    if g == "ridge":
        b = apply_M(m, X)
    else:
        spec = final[g]
        o = torch.load(f"{OUT}/L20/{run_name(dict(DEFAULT, **spec, seed=0))}.pt", weights_only=False)
        net = Net(spec["kind"], m, torch.ones(4096), spec["width"], 0.0, F[20]["s_f"])
        net.load_state_dict({k: v.float() for k, v in o["state"].items()}); net.eval()
        b = net(X)
    print(g, "max rel diff", float((a - b).norm(dim=1).max() / b.norm(dim=1).min()))
