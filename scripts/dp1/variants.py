"""dp1: build variant shard checkpoints from one dp1 fit-shard run (stock dir + '_x<NAME>' siblings).

    python scripts/dp1/variants.py <stock_shard_dir> <out_dir> <K>

Writes <out_dir>/<V>/experts_K<K>_checkpoint.pt (+ position_records.json), each a stock-format shard that the
unmodified merge-experts reads. Row sums per (layer, expert):
  V0  = G                      stock rows
  V1  = G - S                  skip path (computed) removed from the same-position term
  V2  = G - (D1+D2)/2          later positions only (same-position block removed, random-sign estimate)
  V2a = G - D1, V2b = G - D2   one draw each (noise check)
  V3  = G - n_e * I            post-hoc, c = 1 (the formula holds exactly one identity per source position)
  V4  = C                      content targets only
  V5  = C - SC                 content targets, skip path removed
Also writes <out_dir>/checks.json: per layer, max|S/n - I|, max|SC/n - share_e I|, ||V1-V3||/||V0|| (pooled),
tr/d of pooled V0..V5 and of D (pooled same-position block).
"""
import json
import os
import sys

import torch as t

src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
fn = f"experts_K{K}_checkpoint.pt"
ld = lambda d: t.load(os.path.join(d, fn), map_location="cpu", weights_only=True)
base = ld(src)
X = {n: ld(f"{src}_x{n}") for n in ["D1", "D2", "C", "S", "SC"]}
rec = open(os.path.join(src, "position_records.json")).read()
layers = sorted(base["layer_sums"].keys()) if "layer_sums" in base else None
print("keys", [k for k in base.keys() if k != "position_records"])


def sums(state):  # layer -> row sums [E,F,N]
    return {int(l): v["weighted_jacobian_row_sum_EFN"] for l, v in state["layer_sums"].items()}


G, D1, D2, C, S, SC = (sums(s) for s in [base, X["D1"], X["D2"], X["C"], X["S"], X["SC"]])
cnt = {int(l): v["weight_sum_E"] for l, v in base["layer_sums"].items()}
for n, s in X.items():  # counts must agree
    for l, v in s["layer_sums"].items():
        assert t.equal(v["weight_sum_E"], base["layer_sums"][l]["weight_sum_E"]), (n, l)
eye = lambda d: t.eye(d)
V = {
    "V0": lambda l: G[l],
    "V1": lambda l: G[l] - S[l],
    "V2": lambda l: G[l] - 0.5 * (D1[l] + D2[l]),
    "V2a": lambda l: G[l] - D1[l],
    "V2b": lambda l: G[l] - D2[l],
    "V3": lambda l: G[l] - cnt[l][:, None, None] * eye(G[l].shape[-1])[None],
    "V4": lambda l: C[l],
    "V5": lambda l: C[l] - SC[l],
}
checks = {"layers": {}}
for name, f in V.items():
    st = dict(base)
    st["layer_sums"] = {l: dict(v, weighted_jacobian_row_sum_EFN=f(int(l))) for l, v in base["layer_sums"].items()}
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, fn), "wb") as fh:
        t.save(st, fh)
    with open(os.path.join(d, "position_records.json"), "w") as fh:
        fh.write(rec)
    print("wrote", d, flush=True)
for l in sorted(G):
    n = cnt[l].sum()
    pooled = {k: f(l).sum(0) / n for k, f in V.items()}
    dmod = pooled["V0"].shape[0]
    I = eye(dmod)
    Sbar = S[l].sum(0) / n
    nT = None
    c = {
        "max_abs_S_minus_I": float((Sbar - I).abs().max()),
        "S_offdiag_max": float((Sbar - t.diag(t.diag(Sbar))).abs().max()),
        "S_diag_min_max": [float(t.diag(Sbar).min()), float(t.diag(Sbar).max())],
        "SC_diag_mean": float(t.diag(SC[l].sum(0) / n).mean()),
        "SC_offdiag_max": float((SC[l].sum(0) / n - t.diag(t.diag(SC[l].sum(0) / n))).abs().max()),
        "relF_V1_V3": float((pooled["V1"] - pooled["V3"]).norm() / pooled["V0"].norm()),
        "relF_V2a_V2b": float((pooled["V2a"] - pooled["V2b"]).norm() / pooled["V2"].norm()),
        "relF_V2_V0": float((pooled["V2"] - pooled["V0"]).norm() / pooled["V0"].norm()),
        "tr_over_d": {k: float(t.trace(v) / dmod) for k, v in pooled.items()},
        "tr_over_d_D": float(t.trace(0.5 * (D1[l] + D2[l]).sum(0) / n) / dmod),
        "relF_D_over_V0": float((0.5 * (D1[l] + D2[l]).sum(0) / n).norm() / pooled["V0"].norm()),
    }
    checks["layers"][l] = c
    print(l, json.dumps(c), flush=True)
json.dump(checks, open(os.path.join(out, "checks.json"), "w"), indent=1)
