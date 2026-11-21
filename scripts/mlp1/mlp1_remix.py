"""mlp1 secondary analysis (outside the brief's lam grid): unit(Jgen h) + lam * unit(f(h)) for lam in
{0.1, 0.3, 1, 3, 10}, for ridge A and every seed of the final groups at all layers, from the saved states.
lam chosen on validation words (any-of-7 top-10); test any-of-7 reported. Writes results/remix.json.
Usage: mlp1_remix.py FINAL.json"""
import json, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa
from mlp1_train import DEFAULT, Net, run_name, fwd_rows

torch.set_grad_enabled(False)
LX = [0.1, 0.3, 1.0, 3.0, 10.0]
final = json.load(open(sys.argv[1]))
gpu_setup()
D = Data()
JG = lensd(JGEN)
VR, TR = {}, {}
for l in READ:
    X = D.load_X(l).to(dev)
    m = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(f"{OUT}/ridge_L{l}.pt").items()}
    zs = torch.cat([(D.rows(X, D.TRN[i:i + 8192]) - m["mx"]) @ m["P"] for i in range(0, 40960, 8192)])
    s_f = (zs @ m["Wr"] + m["my"]).pow(2).mean().sqrt().item()
    jv, jt = unit(D.rows(X, D.VAL) @ JG[l].T), unit(D.rows(X, D.TE) @ JG[l].T)
    n = 0

    def process(key, fv, ft):
        for lam in LX:
            VR.setdefault((key, lam), {})[l] = D.ranks_of(jv + lam * unit(fv), D.wi[D.VAL]).cpu()
            TR.setdefault((key, lam), {})[l] = D.ranks_of(jt + lam * unit(ft), D.wi[D.TE]).cpu()
    process(("A", 0), apply_M(m, D.rows(X, D.VAL)), apply_M(m, D.rows(X, D.TE)))
    for g, spec in final.items():
        for seed in (0, 1, 2):
            try:
                o = torch.load(f"{OUT}/L{l}/{run_name(dict(DEFAULT, **spec, seed=seed))}.pt", weights_only=False)
            except FileNotFoundError:
                continue
            net = Net(spec["kind"], m, torch.ones(D.d, device=dev), spec["width"], 0.0, s_f).to(dev)
            net.load_state_dict({k: v.float() for k, v in o["state"].items()})
            net.eval()
            process((g, seed), fwd_rows(net, D, X, D.VAL), fwd_rows(net, D, X, D.TE))
            n += 1
            del o, net
    log(f"L{l} done ({n} maps)")
    del X
    torch.cuda.empty_cache()
res = {}
keys = sorted({k for k, _ in VR}, key=str)
for key in keys:
    if any(l not in VR[(key, LX[0])] for l in READ):
        continue
    v = {lam: t10(any7(VR[(key, lam)])) for lam in LX}
    t = {lam: t10(any7(TR[(key, lam)])) for lam in LX}
    lam = max(LX, key=lambda z: (round(v[z], 6), -z))
    res[f"{key[0]}|s{key[1]}"] = dict(val_any7=v, test_any7=t, lam=lam, test_at_lam=t[lam])
json.dump(res, open(f"{RES}/remix.json", "w"), indent=1)
log("REMIX_DONE")
