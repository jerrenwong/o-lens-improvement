"""mlp1 diagnostics at one layer (validation words only): what do the trained maps change relative to ridge A?
 - C decomposition: C's own linear part (z W + b) alone; full C; linear part + the MLP branch replaced by its mean over
   train sentences (an input-independent shift).  If the constant shift recovers most of C's gain, the gain is a
   prior/bias effect, not input-dependent nonlinearity.
 - hubness: share of val sentences whose top-1 token is one of the 20 most frequent top-1 tokens; distinct top-1 tokens.
 - examples: 20 random val sentences with word, A / B / C / D top-6 tokens and ranks.
Usage: MLP1_DEV=cpu|cuda mlp1_diag.py LAYER OUT.json runB runC runD [more run names]"""
import json, random, sys
from collections import Counter
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa
from mlp1_train import Net, fwd_rows

torch.set_grad_enabled(False)
l, outp, names = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
if dev == "cuda":
    gpu_setup()
else:
    torch.set_num_threads(32)
D = Data()
X = D.load_X(l)
m = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(f"{OUT}/ridge_L{l}.pt").items()}
zs = torch.cat([(D.rows(X, D.TRN[i:i + 8192]) - m["mx"]) @ m["P"] for i in range(0, 40960, 8192)])
s_f = (zs @ m["Wr"] + m["my"]).pow(2).mean().sqrt().item()
VAL = D.VAL
xv = D.rows(X, VAL)
wv = D.wi[VAL]
outs = {"A": apply_M(m, xv)}
nets = {}
for n in names:
    o = torch.load(f"{OUT}/L{l}/{n}.pt", weights_only=False)
    sp = o["spec"]
    net = Net(sp["kind"], m, torch.ones(D.d, device=dev), sp["width"], 0.0, s_f).to(dev)
    net.load_state_dict({k: v.float() for k, v in o["state"].items()})
    net.eval()
    nets[n] = net
    outs[n] = torch.cat([net(xv[i:i + 2048]) for i in range(0, len(xv), 2048)])
res = dict(layer=l, val10={k: round(t10(D.ranks_of(v, wv)), 4) for k, v in outs.items()})
# C / L decomposition
for n, net in nets.items():
    if net.kind not in ("C", "L"):
        continue
    act = torch.relu if net.kind == "C" else (lambda t: t)

    def branch(x):
        z = (x - net.mx) @ net.P
        return net.s_f * net.Bo(act(net.A(z / net.zstd)))

    def lin(x):
        return net.b + ((x - net.mx) @ net.P) @ net.W
    xt = D.rows(X, D.TRN[torch.randperm(len(D.TRN), generator=torch.Generator().manual_seed(3))[:8192].to(dev)])
    bmean = branch(xt).mean(0)
    bv = branch(xv)
    res[f"decomp|{n}"] = dict(
        linear_part=round(t10(D.ranks_of(lin(xv), wv)), 4),
        linear_plus_mean_branch=round(t10(D.ranks_of(lin(xv) + bmean, wv)), 4),
        full=round(t10(D.ranks_of(lin(xv) + bv, wv)), 4),
        branch_rms_over_out_rms=round(float(bv.pow(2).mean().sqrt() / (lin(xv) + bv).pow(2).mean().sqrt()), 4),
        branch_var_share=round(float((bv - bv.mean(0)).pow(2).sum() / bv.pow(2).sum()), 4))
# hubness
for k, v in outs.items():
    top1 = torch.cat([D.logits(v[i:i + 256]).argmax(1) for i in range(0, len(v), 256)]).tolist()
    c = Counter(top1)
    res.setdefault("hub", {})[k] = dict(distinct_top1=len(c), top20_share=round(sum(x for _, x in c.most_common(20)) / len(top1), 4),
                                        most_common=[(D.strs[t].strip(), x) for t, x in c.most_common(8)])
# examples
rows = random.Random(11).sample(range(len(VAL)), 20)
ex = []
for j in rows:
    i = int(VAL[j])
    e = dict(word=D.words_s[i], style=int(D.sty[i]) + 1, sent=D.meta["sents"][i])
    for k, v in outs.items():
        lg = D.logits(v[j:j + 1])[0]
        e[k] = [D.strs[t].strip() for t in lg.topk(6).indices.tolist()]
        e[k + "_rank"] = int(D.ranks_of(v[j:j + 1], wv[j:j + 1])[0])
    ex.append(e)
res["examples"] = ex
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
print("DIAG_DONE")
