"""mlp1 export: pack the seed-0 trained map of one final group at all 7 layers into .hf27/mlp1/F_<group>.pt
({layer: dict(kind, s_f, state)}) for the benchmark candidates; also F_ridge.pt from ridge_L*.pt (kind "R").
s_f is recomputed exactly as in mlp1_train.main (first 40,960 train rows). Usage: mlp1_export.py FINAL.json"""
import json, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa
from mlp1_train import DEFAULT, Net, run_name, fwd_rows

torch.set_grad_enabled(False)
if dev == "cpu":
    torch.set_num_threads(32)
final = json.load(open(sys.argv[1]))
D = Data()
packs = {g: {} for g in final}
ridge = {}
for l in READ:
    m = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(f"{OUT}/ridge_L{l}.pt").items()}
    X = D.load_X(l)
    zs = torch.cat([(D.rows(X, D.TRN[i:i + 8192]) - m["mx"]) @ m["P"] for i in range(0, 40960, 8192)])
    s_f = (zs @ m["Wr"] + m["my"]).pow(2).mean().sqrt().item()
    ridge[l] = dict(kind="R", s_f=s_f, state=dict(mx=m["mx"].cpu(), P=m["P"].half().cpu(), W=m["Wr"].cpu(),
                                                  b=m["my"].cpu()))
    for g, spec in final.items():
        o = torch.load(f"{OUT}/L{l}/{run_name(dict(DEFAULT, **spec, seed=0))}.pt", weights_only=False)
        packs[g][l] = dict(kind=spec["kind"], s_f=s_f, state=o["state"], spec=spec)
        # check: fp16-stored state reproduces the stored validation ranks
        net = Net(spec["kind"], m, torch.ones(D.d, device=dev), spec["width"], 0.0, s_f).to(dev)
        net.load_state_dict({k: v.float() for k, v in o["state"].items()})
        net.eval()
        sub = D.VAL[:1000]  # first 1000 validation rows (CPU-friendly check)
        r = D.ranks_of(fwd_rows(net, D, X, sub), D.wi[sub]).cpu()
        log(f"L{l} {g}: val10 (1000 rows) stored {t10(o['VR'][:1000]):.4f} reloaded {t10(r):.4f}; same-rank frac "
            f"{float((r == o['VR'][:1000]).float().mean()):.3f}")
    log(f"L{l} s_f {s_f:.3e}")
    del X, zs
for g, p in packs.items():
    torch.save(p, f"{OUT}/F_{g}.pt")
torch.save(ridge, f"{OUT}/F_ridge.pt")
log("EXPORT_DONE")
