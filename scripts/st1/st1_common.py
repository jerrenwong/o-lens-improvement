"""st1 shared helpers (same definitions as scripts/st1/st1_fit.py, for st1_p3.py / st1_qual.py)."""
import json
import re

import torch

R = "/workspace/o-lens-improvement"
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
dev = "cuda"
N = len(json.load(open(f"{R}/runs/e1/prefixes.json")))
model = Wu = V = d = gamma = strs = nonword = func = mask = NORM = junk_tok = None


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def setup(m):
    global model, Wu, V, d, gamma, strs, nonword, func, mask, NORM, junk_tok
    model = m
    tok = m.tokenizer
    Wu = m._lm_head.weight
    V, d = Wu.shape
    strs = [tok.decode([i]) for i in range(V)]
    nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
    func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
    mask = nonword | func
    norm_s = [s.strip().lower() for s in strs]
    vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
    NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
    junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
    junk_tok[[i for i in special if i < V]] = True
    junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)
    norm = m._final_norm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)) or 0.0)
    with torch.no_grad():
        gamma = norm(torch.ones(1, d, device=dev)).float()[0] * (1.0 + eps) ** 0.5


@torch.no_grad()
def logits(v, bs=250):
    return torch.cat([model.unembed(v[i:i + bs].to(dev).float()).float() for i in range(0, v.shape[0], bs)])


@torch.no_grad()
def topk(v, k=10, bs=250, m="default"):
    m = mask if isinstance(m, str) else m
    out = []
    for i in range(0, v.shape[0], bs):
        lg = model.unembed(v[i:i + bs].to(dev).float()).float()
        if m is not None:
            lg = lg.masked_fill(m, float("-inf"))
        out.append(lg.topk(k, -1).indices)
    return torch.cat(out)


def agree(top, word):
    return (NORM[top] == NORM[word][:, None]).any(-1)


@torch.no_grad()
def diag(pred, wt):
    tu = topk(pred, 10, m=None)
    tm = topk(pred, 10)
    nw = nonword | junk_tok
    return dict(nw1=nw[tu[:, 0]].float().mean().item(), nw10=nw[tu].float().mean().item(),
                fn1=func[tu[:, 0]].float().mean().item(),
                a1u=(NORM[tu[:, 0]] == NORM[wt]).float().mean().item(),
                a1=agree(tm[:, :1], wt).float().mean().item(), a10=agree(tm, wt).float().mean().item())


def avg_rank(x):
    o = x.argsort(); r = torch.empty_like(x); r[o] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    u, inv = torch.unique(x, return_inverse=True)
    s = torch.zeros(len(u), device=x.device, dtype=x.dtype).scatter_add_(0, inv, r)
    c = torch.zeros(len(u), device=x.device, dtype=x.dtype).scatter_add_(0, inv, torch.ones_like(r))
    return (s / c)[inv]


@torch.no_grad()
def decode_check(name, tgt, ids, val, keep, rows, quiet=False, sp_n=None):
    lg = logits(tgt[rows]).masked_fill(~keep, float("-inf"))
    lq = torch.log_softmax(lg, -1)
    I, P = ids[rows].long().to(dev), val[rows].to(dev)
    sup = P > 0
    kl = (torch.where(sup, P * (P.clamp_min(1e-30).log() - lq.gather(1, I)), torch.zeros_like(P))).sum(-1).mean().item()
    t1 = (lg.argmax(-1) == I[:, 0]).float().mean().item()
    d10, d50 = lg.topk(10, -1).indices, lg.topk(50, -1).indices
    nsup = sup.sum(-1).clamp(max=10)
    ov = torch.stack([(torch.isin(d10[j], I[j, :nsup[j]])).sum() / nsup[j] for j in range(len(rows))]).float().mean().item()
    sc = []
    for j in range(len(rows) if sp_n is None else min(sp_n, len(rows))):
        u = torch.unique(torch.cat([I[j][sup[j]], d50[j]]))
        pv = torch.zeros(V, device=dev); pv[I[j][sup[j]]] = P[j][sup[j]]
        a, b = avg_rank(lg[j, u]), avg_rank(pv[u])
        if b.std() > 0:
            sc.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    sp = sum(sc) / max(1, len(sc))
    if not quiet:
        print(f"   decode check {name:14s}: KL(p||q) {kl:.3f}; Spearman {sp:.3f}; top-1 kept {t1:.3f}; "
              f"top-10 overlap {ov:.3f} (n={len(rows)})", flush=True)
    return dict(kl=kl, spearman=sp, top1=t1, overlap10=ov)
