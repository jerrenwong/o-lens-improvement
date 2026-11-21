"""mlp1 precision check: all ranks elsewhere use bf16 logits ((v*g).bf16 @ U.T, as al2/mp1) with rank = 1 + #(logit >
best form), so bf16 ties count in the target's favour. Here: test ranks with fp32 logits (U upcast in vocab chunks), for
ridge A and the seed-0 final maps, alone and mixed with Jgen (lam 0.3 and 1.0), any-of-7 top-10 and top-1.
Usage: mlp1_fp32check.py FINAL.json"""
import json, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa
from mlp1_train import DEFAULT, Net, run_name, fwd_rows

torch.set_grad_enabled(False)
final = json.load(open(sys.argv[1]))
gpu_setup()
D = Data()
JG = lensd(JGEN)
VC = 32768


def ranks32(vecs, wid, bs=512):
    out = []
    for i in range(0, vecs.shape[0], bs):
        v = vecs[i:i + bs].float() * D.g
        lg = torch.cat([v @ D.U[j:j + VC].float().T for j in range(0, D.V, VC)], 1).masked_fill_(D.excl, float("-inf"))
        best = lg.gather(1, D.FORM[wid[i:i + bs]]).max(1).values
        out.append(torch.count_nonzero(lg > best[:, None], dim=1) + 1)
    return torch.cat(out).cpu()


TR = {}
for l in READ:
    X = D.load_X(l).to(dev)
    m = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(f"{OUT}/ridge_L{l}.pt").items()}
    zs = torch.cat([(D.rows(X, D.TRN[i:i + 8192]) - m["mx"]) @ m["P"] for i in range(0, 40960, 8192)])
    s_f = (zs @ m["Wr"] + m["my"]).pow(2).mean().sqrt().item()
    xt = D.rows(X, D.TE)
    jt = unit(xt @ JG[l].T)
    wt = D.wi[D.TE]

    def process(name, ft):
        TR.setdefault(name, {})[l] = ranks32(ft, wt)
        for lam in (0.3, 1.0):
            TR.setdefault(f"mix{lam}|{name}", {})[l] = ranks32(jt + lam * unit(ft), wt)
    process("A", apply_M(m, xt))
    TR.setdefault("Jgen", {})[l] = ranks32(xt @ JG[l].T, wt)
    for g, spec in final.items():
        o = torch.load(f"{OUT}/L{l}/{run_name(dict(DEFAULT, **spec, seed=0))}.pt", weights_only=False)
        net = Net(spec["kind"], m, torch.ones(D.d, device=dev), spec["width"], 0.0, s_f).to(dev)
        net.load_state_dict({k: v.float() for k, v in o["state"].items()})
        net.eval()
        process(g, torch.cat([net(xt[i:i + 2048]) for i in range(0, len(xt), 2048)]))
        del o, net
    log(f"L{l} done")
    del X
    torch.cuda.empty_cache()
res = {n: dict(any7_top10=round(t10(any7(v)), 4), any7_top1=round(t1(any7(v)), 4),
               L20_top10=round(t10(v[20]), 4)) for n, v in TR.items()}
json.dump(res, open(f"{RES}/fp32check.json", "w"), indent=1)
torch.save(TR, f"{OUT}/fp32_test_ranks.pt")
log("FP32_DONE")
