"""w11 qualitative (FIT split only): J-lens vs prob-J top-10 for 10 fit items (2 per task) at prob-J's best single
FIT layer, f4 display (masked ids removed, one token per word group). Writes runs/w11/qual_<tag>.md, then returns the
prob-J lens (so the evaluator run doubles as a dev run; name it w11_qual_*). Config: W11_* env as in w11_lib."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w11_lib as W  # noqa: E402
import w11c_m3lens as WB  # noqa: E402
from fm27_lib import NEG, excl_mask  # noqa: E402


def build(ctx):
    lens = WB.build(ctx)
    tag = W.TAG[ctx.model.d_model]
    mask = excl_mask(ctx).to(ctx.device)
    pl = W.per_layer_fit(ctx, lens, mask)
    best = max(pl, key=pl.get)
    jl = W.ProbJ.__new__(W.ProbJ)
    jl.model, jl.base, jl.mode, jl.beta, jl.reg = ctx.model, W.make_base(ctx, "J"), "base", 0.0, {}
    plj = W.per_layer_fit(ctx, jl, mask)
    tok = ctx.model.tokenizer
    group_topk = sys.modules["w1_filters"].group_topk
    gid = ctx.fit.group_ids
    seen, pick = {}, []
    for i, e in enumerate(ctx.fit.eval_slugs):
        if seen.get(e, 0) < 2:
            seen[e] = seen.get(e, 0) + 1; pick.append(i)
    out = [f"# w11 qualitative, {tag}, FIT items, layer {best} (prob-J's best single FIT layer)\n",
           f"per-layer FIT p10+p1: prob-J {({k: round(v, 3) for k, v in pl.items()})}",
           f"J-lens {({k: round(v, 3) for k, v in plj.items()})}\n",
           f"M3 config: W{os.environ.get('W11C_W')} tn {os.environ.get('W11C_TN')} form {os.environ.get('W11C_FORM')}\n"]
    h = ctx.fit.resid[best][pick]
    LJ = jl.logits(h, best).float().masked_fill(mask, NEG)
    LP = lens.logits(h, best).float().masked_fill(mask, NEG)
    for r, i in enumerate(pick):
        labels = {k: [tok.decode([t]) for t in v] for k, v in ctx.fit.candidates[i].items()}
        out.append(f"## {ctx.fit.eval_slugs[i]} / {ctx.fit.item_names[i]}\nlabels: {labels}")
        for name, L in (("J-lens", LJ), ("prob-J", LP)):
            ids = group_topk(L[r], gid, 10) if gid is not None else L[r].topk(10).indices.tolist()
            out.append(f"- {name}: " + " | ".join(repr(tok.decode([t])) for t in ids))
        out.append("")
    Path(f"{W.R}/runs/w11/qual_w11c_{tag}_W{os.environ.get("W11C_W", "1")}_tn{os.environ.get("W11C_TN", "0")}.md").write_text("\n".join(out))
    return lens
