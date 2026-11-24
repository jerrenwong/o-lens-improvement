"""w9 targets: pluggable answer-distribution targets for the cross-entropy maps.
Every builder has the signature  build(C, Yk, wk, K, T, rows=None) -> (ids [n, 3K] int, val [n, 3K] float)
  C   : scripts/st1/st1_common after C.setup(model) (unembed, junk rules)
  Yk  : list of 3 [N, d] unit-normalised answer states (wordings), wk [3, N] wording weights (junk-dropped wordings 0)
TARGETS = {"w2": verbatim scripts/w2/w2_ce_fit.build_p (top-K word tokens per wording, softmax, average),
           "w7nomerge": scripts/w7/w7_fit.py mode nomerge (f4 token rule, softmax over allowed tokens, top K,
                        renormalise, average) -- the w7 agent's refined target; swap by --target.}
To add a target: write build_<name>(C, Yk, wk, K, T, rows) and register it in TARGETS."""
import importlib.util

import torch

R = "/workspace/o-lens-improvement"
dev = "cuda"


@torch.no_grad()
def _accumulate(C, Yk, wk, K, rows, per_wording, bs=250):
    N = Yk[0].shape[0]
    rows = torch.arange(N) if rows is None else rows
    ids_all, val_all = [], []
    for s0 in range(0, len(rows), bs):
        r = rows[s0:s0 + bs]
        D = torch.zeros(len(r), C.V, device=dev)
        for k in range(3):
            ii, pk = per_wording(C.logits(Yk[k][r]))
            D.scatter_add_(1, ii, pk * wk[k, r.to(dev), None])
        vv, ii = D.topk(3 * K, -1)
        ids_all.append(ii.int().cpu()); val_all.append(vv.cpu())
    return torch.cat(ids_all), torch.cat(val_all)


def build_w2(C, Yk, wk, K=10, T=1.0, rows=None):
    keep = ~C.nonword & ~C.junk_tok

    def pw(lg):
        vv, ii = lg.masked_fill(~keep, float("-inf")).topk(K, -1)
        return ii, torch.softmax(vv / T, -1)
    return _accumulate(C, Yk, wk, K, rows, pw)


_ALLOWED = {}


def _allowed(C):
    if "a" not in _ALLOWED:
        fs = importlib.util.spec_from_file_location("w9_filters", f"{R}/auto/filters.py")
        FM = importlib.util.module_from_spec(fs); fs.loader.exec_module(FM)
        SPEC = FM.special_ids(C.model.tokenizer)
        bad = ("@",) + tuple(FM.MARKUP_CHARS)
        _ALLOWED["a"] = torch.tensor([any(c.isalnum() for c in s) and i not in SPEC and not any(c in s for c in bad)
                                      for i, s in enumerate(C.strs)], device=dev)
    return _ALLOWED["a"]


def build_w7nomerge(C, Yk, wk, K=10, T=1.0, rows=None):
    al = _allowed(C)

    def pw(lg):
        p = torch.softmax(lg.masked_fill(~al, float("-inf")) / T, -1)
        vv, ii = p.topk(K, -1)
        return ii, vv / vv.sum(-1, keepdim=True)
    return _accumulate(C, Yk, wk, K, rows, pw)


TARGETS = {"w2": build_w2, "w7nomerge": build_w7nomerge}
