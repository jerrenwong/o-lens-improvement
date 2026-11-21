"""mlp1 training: per-layer maps decoded through frozen g and U exactly like ridge M, trained with full-vocabulary
cross-entropy (target = the word's single-token forms, loss = logsumexp(all non-excluded) - logsumexp(forms)),
early-stopped on validation WORDS (best val top-10; eval every 20 steps up to step 200, then every EVAL=100;
stop after 10*EVAL steps without improvement; max 20 epochs).
  B: f(h) = z W + b                         (W, b init = ridge M)                 -> objective only
  C: f(h) = z W + b + Bo relu(A u + a)      (W, b init = ridge M, Bo = 0)         -> + nonlinearity
  D: f(h) = Bo relu(A u + a) + b            (default init, no linear skip)       -> from scratch
  L: as C without the ReLU (linear control with C's parametrisation / learning rates)
  P: ridge M frozen + trained output shift s_f * c (input-independent prior/bias control)
  Q: B + trained output shift s_f * c (c at lr LR_C = P's best lr): objective + prior shift, still linear
z = (h - mx) @ P is the ridge recipe's whitened input (P from within-word covariance), u = z / std_train(z) per dim;
input dropout on z. A learnable logit scale tau (log-parametrised, init by a 1-D search on train CE, lr 1e-2, no wd)
multiplies the logits in the loss only (ranks and unit() mixes are scale-free).
Usage: mlp1_train.py LAYER SPECS.json   (list of spec dicts; finished runs are skipped)"""
import copy, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa

BS, CH, EVAL, PATIENCE, EPOCHS = 1024, 256, 100, 10, 20
EPOCHS = int(os.environ.get("MLP1_EPOCHS", EPOCHS))  # step-cap check (default 20)
if os.environ.get("MLP1_TINY"):  # quick test: 1 epoch, eval every 20 steps, max 60 steps
    EVAL, EPOCHS = 20, 1
LR_C = 1e-2
DEFAULT = dict(kind="B", width=0, lr=1e-4, wd=0.0, drop=0.0, seed=0, data="sent", noise=0.0, subset="all", save=True)


def run_name(s):
    w = s["width"] if s["kind"] != "B" else ""
    dat = "sent" if s["data"] == "sent" else f"mean{s['noise']}"
    return f"{s['kind']}{w}_lr{s['lr']:g}_wd{s['wd']:g}_dr{s['drop']:g}_{dat}_{s['subset']}_s{s['seed']}"


class Net(nn.Module):
    def __init__(self, kind, m, zstd, width, drop, s_f):
        super().__init__()
        self.kind, self.drop, self.s_f = kind, drop, s_f
        self.register_buffer("mx", m["mx"].float())
        self.register_buffer("P", m["P"].float())
        self.register_buffer("zstd", zstd)
        d = m["P"].shape[0]
        if kind in ("B", "C", "L", "Q"):
            self.W = nn.Parameter(m["Wr"].float().clone())
            self.b = nn.Parameter(m["my"].float().clone())
            if kind == "Q":
                self.c = nn.Parameter(torch.zeros(d))
        elif kind == "P":  # ridge map frozen, only an output shift b0 + s_f * c is trained (prior / bias control)
            self.register_buffer("W", m["Wr"].float().clone())
            self.register_buffer("b0", m["my"].float().clone())
            self.c = nn.Parameter(torch.zeros(d))
        else:
            self.b = nn.Parameter(torch.zeros(d))
        if kind in ("C", "D", "L"):
            self.A = nn.Linear(d, width)
            self.Bo = nn.Linear(width, d, bias=False)
            if kind in ("C", "L"):
                nn.init.zeros_(self.Bo.weight)
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, h, noise=0.0):
        z = (h - self.mx) @ self.P
        if noise and self.training:
            z = z + noise * torch.randn_like(z)
        if self.drop and self.training:
            z = F.dropout(z, self.drop)
        if self.kind == "P":
            return z @ self.W + self.b0 + self.s_f * self.c
        if self.kind == "D":
            return self.s_f * (self.b + self.Bo(F.relu(self.A(z / self.zstd))))
        out = self.b + z @ self.W
        if self.kind == "Q":
            out = out + self.s_f * self.c
        if self.kind == "C":
            out = out + self.s_f * self.Bo(F.relu(self.A(z / self.zstd)))
        elif self.kind == "L":  # C without the ReLU (linear control with C's parametrisation and learning rates)
            out = out + self.s_f * self.Bo(self.A(z / self.zstd))
        return out


def ce(D, v, w, log_tau):
    lg = D.logits(v, log_tau.exp())
    pos = lg.gather(1, D.FORM[w]).masked_fill(~D.FVALID[w], float("-inf")).logsumexp(1)
    return lg.logsumexp(1) - pos


@torch.no_grad()
def fwd_rows(net, D, X, idx, bs=2048):
    return torch.cat([net(D.rows(X, idx[i:i + bs])) for i in range(0, len(idx), bs)])


def train_one(D, X, l, s, m, zstd, jg, Mn_tr, tr_w, idx_tr):
    torch.manual_seed(s["seed"])
    net = Net(s["kind"], m, zstd, s["width"], s["drop"], m["s_f"]).to(dev)
    # tau init: 1-D search on 4096 train rows (no grad)
    net.eval()
    g0 = torch.Generator(device="cpu").manual_seed(7)
    probe = idx_tr[torch.randperm(len(idx_tr), generator=g0)[:4096].to(dev)]
    with torch.no_grad():
        if s["data"] == "sent":
            vp, wp = fwd_rows(net, D, X, probe), D.wi[probe]
        else:
            vp, wp = net(Mn_tr[:4096]), tr_w[:4096]
        best = None
        for lt in torch.linspace(-8, 8, 65):
            lt = lt.to(dev)
            L = torch.cat([ce(D, vp[i:i + 512], wp[i:i + 512], lt) for i in range(0, len(vp), 512)]).mean().item()
            if best is None or L < best[0]:
                best = (L, lt.item())
        net.log_tau.fill_(best[1])
    # Adam steps are ~lr per entry, so the ridge-initialised W, b get lr scaled by their own rms (= relative lr);
    # the MLP branch is in units of s_f (rms of ridge outputs) and uses the plain lr.
    lr = s["lr"]
    groups = [dict(params=[net.log_tau], lr=1e-2, weight_decay=0.0)]
    if s["kind"] == "P":
        groups += [dict(params=[net.c], lr=lr, weight_decay=0.0)]
    elif s["kind"] in ("B", "C", "L", "Q"):
        if s["kind"] == "Q":
            groups += [dict(params=[net.c], lr=LR_C, weight_decay=0.0)]
        groups += [dict(params=[net.W], lr=lr * m["Wr"].pow(2).mean().sqrt().item(), weight_decay=s["wd"]),
                   dict(params=[net.b], lr=lr * m["my"].pow(2).mean().sqrt().item(), weight_decay=0.0)]
    elif s["kind"] == "D":
        groups += [dict(params=[net.b], lr=lr, weight_decay=0.0)]
    if s["kind"] in ("C", "D", "L"):
        groups += [dict(params=[net.A.weight, net.Bo.weight], lr=lr, weight_decay=s["wd"]),
                   dict(params=[net.A.bias], lr=lr, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(1000 + s["seed"])
    n_ex = len(idx_tr) if s["data"] == "sent" else len(tr_w)
    steps_ep = math.ceil(n_ex / BS)
    max_steps = EPOCHS * math.ceil(len(D.TRN) / BS)  # same step budget for all data modes / subsets
    if os.environ.get("MLP1_TINY"):
        max_steps = 60
    curve, best, bad, step, lsum, lcnt = [], None, 0, 0, 0.0, 0

    def evaluate():
        net.eval()
        rv = D.ranks_of(fwd_rows(net, D, X, D.VAL), D.wi[D.VAL])
        rs = D.ranks_of(fwd_rows(net, D, X, D.TRS), D.wi[D.TRS])
        net.train()
        return (round(t10(rv), 6), t1(rv)), t10(rs)

    z, zs = evaluate()
    curve.append(dict(step=0, val10=z[0], val1=z[1], tr10=zs, loss=None, tau=net.log_tau.exp().item()))
    best = (z, 0, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}))
    t0 = time.time()
    net.train()
    done = False
    while not done:
        perm = torch.randperm(n_ex, generator=gen).to(dev)
        for bi in range(steps_ep):
            b = perm[bi * BS:(bi + 1) * BS]
            opt.zero_grad(set_to_none=True)
            for c in range(0, len(b), CH):
                bc = b[c:c + CH]
                if s["data"] == "sent":
                    rows = idx_tr[bc]
                    h, w = D.rows(X, rows), D.wi[rows]
                else:
                    h, w = Mn_tr[bc], tr_w[bc]
                loss = ce(D, net(h, s["noise"]), w, net.log_tau).sum() / len(b)
                loss.backward()
                lsum += loss.item()
            lcnt += 1
            opt.step()
            step += 1
            if (step % EVAL == 0 or step == max_steps or (step <= 200 and step % 20 == 0)):
                z, zs = evaluate()
                curve.append(dict(step=step, val10=z[0], val1=z[1], tr10=zs, loss=lsum / max(lcnt, 1),
                                  tau=net.log_tau.exp().item()))
                lsum, lcnt = 0.0, 0
                if z > best[0]:
                    best, bad = (z, step, copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})), 0
                else:
                    bad += 1
                if step - best[1] >= PATIENCE * EVAL:  # patience = 10 evaluations at the regular interval
                    done = True
            if step >= max_steps or done:
                done = True
                break
    net.load_state_dict(best[2])
    net.eval()
    with torch.no_grad():
        fv, ft, fs = fwd_rows(net, D, X, D.VAL), fwd_rows(net, D, X, D.TE), fwd_rows(net, D, X, D.TRS)
        out = dict(spec=s, layer=l, curve=curve, best_step=best[1], steps=step, secs=time.time() - t0,
                   tau0=curve[0]["tau"], tau=net.log_tau.exp().item(),
                   VR=D.ranks_of(fv, D.wi[D.VAL]).cpu(), TR=D.ranks_of(ft, D.wi[D.TE]).cpu(),
                   SR=D.ranks_of(fs, D.wi[D.TRS]).cpu())
        for lam in LAMS:
            out[f"VRmix{lam}"] = D.ranks_of(jg["v"] + lam * unit(fv), D.wi[D.VAL]).cpu()
            out[f"TRmix{lam}"] = D.ranks_of(jg["t"] + lam * unit(ft), D.wi[D.TE]).cpu()
    if s["save"]:
        out["state"] = {k: (v.half() if v.is_floating_point() and v.dim() > 0 else v) for k, v in best[2].items()}
    return out


def main():
    l = int(sys.argv[1])
    specs = [dict(DEFAULT, **s) for s in json.load(open(sys.argv[2]))]
    gpu_setup()
    D = Data()
    m = torch.load(f"{OUT}/ridge_L{l}.pt")
    m = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in m.items()}
    X = D.load_X(l).to(dev)  # fp16 on GPU (1 GB); gathering rows on the shared CPU took ~150 ms per 256 rows
    JG = lensd(JGEN)[l]
    with torch.no_grad():
        jg = dict(v=unit(D.rows(X, D.VAL) @ JG.T), t=unit(D.rows(X, D.TE) @ JG.T))
        zs = torch.cat([(D.rows(X, D.TRN[i:i + 8192]) - m["mx"]) @ m["P"] for i in range(0, 40960, 8192)])
        zstd = zs.std(0).clamp_min(1e-4)
        m["s_f"] = (zs @ m["Wr"] + m["my"]).pow(2).mean().sqrt().item()  # rms of ridge outputs per dim (train)
        del zs
    log(f"L{l}: s_f {m['s_f']:.3e}, rms Wr {m['Wr'].pow(2).mean().sqrt().item():.3e}, rms my "
        f"{m['my'].pow(2).mean().sqrt().item():.3e}, zstd mean {zstd.mean().item():.3f}")
    os.makedirs(f"{OUT}/L{l}", exist_ok=True)
    infer = None
    for s in specs:
        name = run_name(s)
        fn = f"{OUT}/L{l}/{name}{os.environ.get('MLP1_SUFFIX', '')}.pt"
        if os.path.exists(fn):
            log(f"skip {name}")
            continue
        idx_tr = D.TRN
        if s["subset"] == "infer":
            if infer is None:
                infer = torch.load(f"{OUT}/infer_rows.pt").to(dev)  # written by mlp1_infer.py
            idx_tr = D.TRN[torch.isin(D.TRN, infer)]
        Mn_tr = tr_w = None
        if s["data"] == "mean":
            with torch.no_grad():
                uw, inv = torch.unique(D.wi[idx_tr], return_inverse=True)
                Mn_tr = torch.zeros(len(uw), D.d, device=dev)
                cnt = torch.zeros(len(uw), device=dev)
                for i in range(0, len(idx_tr), 16384):
                    Mn_tr.index_add_(0, inv[i:i + 16384], D.rows(X, idx_tr[i:i + 16384]))
                    cnt.index_add_(0, inv[i:i + 16384], torch.ones(len(inv[i:i + 16384]), device=dev))
                Mn_tr /= cnt[:, None]
                tr_w = uw
        log(f"start {name} (train rows {len(idx_tr)})")
        out = oom_retry(train_one, D, X, l, s, m, zstd, jg, Mn_tr, tr_w, idx_tr)
        torch.save(out, fn)
        row = dict(name=name, layer=l, **{k: s[k] for k in DEFAULT if k != "save"}, best_step=out["best_step"],
                   steps=out["steps"], secs=round(out["secs"]), val10=round(t10(out["VR"]), 4),
                   val1=round(t1(out["VR"]), 4), test10=round(t10(out["TR"]), 4), test1=round(t1(out["TR"]), 4),
                   train10=round(t10(out["SR"]), 4),
                   **{f"mixval{lam}": round(t10(out[f"VRmix{lam}"]), 4) for lam in LAMS},
                   **{f"mixtest{lam}": round(t10(out[f"TRmix{lam}"]), 4) for lam in LAMS})
        with open(f"{RES}/runs_L{l}{os.environ.get('MLP1_SUFFIX', '')}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"done {name}: val10 {row['val10']} test10 {row['test10']} train10 {row['train10']} "
            f"best_step {row['best_step']}/{row['steps']} ({row['secs']}s)")
        del out
        torch.cuda.empty_cache()
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
