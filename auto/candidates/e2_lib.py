"""Shared code for e2_* candidates (not a candidate itself): model-agnostic copy of e1_lib.
Maps are picked by ctx.model.d_model (4096 -> 9B, 2048 -> 2B) and read from runs/e1/ only.

E_l is a LEARNED PROBE distilled from the model's own answers, not a Jacobian: a per-layer ridge map from the
residual at a prefix's last token (prefix run alone) to the model's final-layer state at the end of three
few-shot one-word elicitation prompts that wrap the same prefix (mean of the three unit-normalised vectors).
Fitted offline, label-free, on 30.5k prefixes of WikiText / TinyStories / 10-language Wikipedia
(scripts/e1_data.py, e1_collect.py, e1_fit.py). Candidates only LOAD the maps.
Lens: unit(JPP_l h) + lam * unit(E_l h + bias), E_l h = (h - mx) @ W.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_common import fit_pass, unit  # noqa: E402

RUNS = "/workspace/o-lens-improvement/runs/e1"
TAGS = {4096: "9b", 2048: "2b"}


def map_path(ctx, variant="main"):
    tag = TAGS[ctx.model.d_model]
    return f"{RUNS}/E_{tag}.pt" if variant == "main" else f"{RUNS}/E_{tag}_{variant}.pt"

LAMS = [0.1, 0.2, 0.3, 0.5, 1.0]


def load_maps(path, dev):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items()}


class ELens:
    """mode 'mix': unit(JPP h) + lam unit(E h + b); 'alone': E h + b; 'prior': unit(JPP h) + lam unit(b)."""

    def __init__(self, JPP, maps, mode, lam=None):
        self.JPP, self.maps, self.mode, self.lam = JPP, maps, mode, lam

    def E(self, h, l):
        mx, W, my = self.maps[l]
        return (h - mx) @ W + my

    def transport(self, h, layer, resid_all=None):
        if self.mode == "alone":
            return self.E(h, layer)
        j = unit(h @ self.JPP[layer].T)
        if self.mode == "prior":
            return j + self.lam * unit(self.maps[layer][2])[None].expand_as(j)
        return j + self.lam * unit(self.E(h, layer))


def fit_vecs(ctx, lens):
    return {l: lens.transport(ctx.fit.resid[l], l, ctx.fit.resid) for l in ctx.readout_layers}


def fit_line(name, ctx, lens):
    p, task = fit_pass(ctx, fit_vecs(ctx, lens))
    print(f"[e1 fit] {name:12s} pass@10 {p:.4f}  " + " ".join(f"{t[:6]} {v:.2f}" for t, v in task.items()),
          flush=True)
    return p


def build_fixed(ctx, path, mode, lam):
    maps = load_maps(path, ctx.device)
    lens = ELens(ctx.JPP, maps, mode, lam)
    fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    fit_line(f"{mode} {lam}", ctx, lens)
    return lens
