"""w3 shared code (not a candidate). "Jacobian A-lens": alens.py with each ridge map replaced by a Jacobian toward the
asked answer (plain dicts {layer: [d, d]} from scripts/ja1/ja1_post.py; full recipe "A5" or label-free pooled "K1").
  ja(h) = unit( unit(J_topic h) + unit(J_unstated h) )        env W3_JT, W3_JU (one of them only -> unit(J h))
  alone: ja(h)                                               (w3_alone.py)
  add:   unit(J++ h) + lam * ja(h)                           (w3_add.py)
lam chosen on the FIT split with alens.py's GRID and rule (fm27_lib.choose: argmax filtered pass@1 + pass@10, ties ->
smaller lam), using alens.py's own helpers (imported, not copied)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alens import GRID  # noqa: E402
from ao1_lib import excl_mask, fmt, ranks, score  # noqa: E402
from fm27_lib import choose, unit  # noqa: E402


def load_Js(ctx):
    out = []
    for var in ("W3_JT", "W3_JU"):
        p = os.environ.get(var)
        if not p:
            continue
        M = torch.load(p, map_location="cpu", weights_only=True)
        M = {int(l): v.to(ctx.device).float() for l, v in M.items()}
        assert next(iter(M.values())).shape[0] == ctx.model.d_model, f"{var} is for another model"
        out.append(M)
    assert out, "set W3_JT and/or W3_JU"
    return out


class JA:
    def __init__(self, Js, base=None, lam=1.0):
        self.Js, self.base, self.lam = Js, base, lam

    def transport(self, h, layer, resid_all=None):
        if self.base is not None and self.lam == 0:
            return unit(h @ self.base[layer].T.to(h))
        parts = [unit(h @ J[layer].T) for J in self.Js]
        a = unit(sum(parts)) if len(parts) > 1 else parts[0]
        if self.base is None:
            return a
        return unit(h @ self.base[layer].T.to(h)) + self.lam * a


def build_alone(ctx):
    lens = JA(load_Js(ctx))
    print(fmt("w3 jacobian A-lens alone FIT", score(ctx, ranks(ctx, lens, excl_mask(ctx)))), flush=True)
    return lens


def build_add(ctx):
    Js, mask, res = load_Js(ctx), excl_mask(ctx), []
    for w in GRID:
        s = score(ctx, ranks(ctx, JA(Js, ctx.JPP, w), mask))
        res.append((w, s))
        print(fmt(f"w3 add lam {w}", s), flush=True)
    lam = choose(res)
    print(f"[w3 add] chosen lam {lam}", flush=True)
    return JA(Js, ctx.JPP, lam)
