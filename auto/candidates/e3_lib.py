"""Shared code for e3_* candidates (not a candidate itself). Maps picked by ctx.model.d_model
(4096 -> 9b, 5120 -> 27b), read from runs/e3/ only (fitted offline by scripts/e3_run.py, ordinary text only).

B_l: per-layer ridge map from the residual at a prefix's last token (centred) to the CENTRED content-word target
t = sum_w p(w) u_w (top-20 content-word tokens of the model's averaged elicited next-token distribution; u_w =
unit-norm unembedding row). No bias. Output returned in the final basis: to_final(t) = t / g (g = final-norm gain),
so model.unembed(to_final(t)) ranks tokens like W_U t.
Lens B: unit(JPP_l h) + lam * unit(to_final(B_l h)).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line  # noqa: E402,F401

RUNS = "/workspace/o-lens-improvement/runs/e3"
TAGS = {4096: "9b", 5120: "27b"}


def map_path(ctx, variant="main"):
    d = ctx.model.d_model
    if d not in TAGS:
        raise RuntimeError(f"e3: no B map for d_model={d}; maps exist only for Qwen3.5-9B (d_model 4096)")
    tag = TAGS[d]
    p = f"{RUNS}/B_{tag}.pt" if variant == "main" else f"{RUNS}/B_{tag}_{variant}.pt"
    if not Path(p).exists():
        raise RuntimeError(f"e3: map {p} not found; B was fitted for Qwen3.5-9B only (27B run cancelled)")
    return p


def load_maps(path, dev):
    """-> {layer: (mx, W_final, 0)} for e2_lib.ELens; W_final = W * scale / g (columns)."""
    m = torch.load(path, map_location="cpu")
    out = {}
    for l, v in m.items():
        W = v["W"].float() * v["scale"] * v["ginv"].float()[None, :]
        mx = v["mx"].float()
        out[l] = (mx.to(dev), W.to(dev), torch.zeros(W.shape[1], device=dev))
    return out


def choose_lam(ctx, maps, tag="B"):
    base = fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"{tag} mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[e3 fit] chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return best
