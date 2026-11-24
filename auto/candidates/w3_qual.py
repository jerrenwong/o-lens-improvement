"""w3 qualitative (FIT split only). 10 fit items (2 per task: first J++-miss and first J++-hit... see pick()), top-10
UNFILTERED tokens of: Jacobian A-lens full recipe (env W3_JT, W3_JU), its label-free K1 version (env W3_JT1, W3_JU1),
and the regression A-lens (alens.py maps), each at its own best FIT layer (max FIT top-10 hit rate, filtered).
'*' = a candidate token, '~' = a token the filtered score would exclude. Returns the full-recipe Jacobian A-lens."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alens import A, R, TAG  # noqa: E402
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import NEG, excl_mask, logits  # noqa: E402
from w3_lib import JA  # noqa: E402


def _load(ctx, p):
    M = torch.load(p, map_location="cpu", weights_only=True)
    return {int(l): v.to(ctx.device).float() for l, v in M.items()}


def build(ctx):
    tok, f, mask = ctx.model.tokenizer, ctx.fit, excl_mask(ctx)
    tag = TAG[ctx.model.d_model]
    E = load_maps(f"{R}/.hf27/ask3/E_{tag}_t3.pt", ctx.device)
    E2 = load_maps(f"{R}/.hf27/ask5/E_{tag}_s5.pt", ctx.device)
    lenses = {"JacA full": JA([_load(ctx, os.environ["W3_JT"]), _load(ctx, os.environ["W3_JU"])]),
              "JacA K1": JA([_load(ctx, os.environ["W3_JT1"]), _load(ctx, os.environ["W3_JU1"])]),
              "regA": A(None, E, E2, 1.0), "J++": JA([ctx.JPP])}
    excl = set(f.excluded_ids.tolist())
    top_u, hit = {}, {}
    for name, lens in lenses.items():
        for l in ctx.readout_layers:
            lg = logits(ctx.model, lens.transport(f.resid[l], l))
            top_u[name, l] = lg.topk(10, -1).indices.cpu()
            tf = lg.masked_fill(mask, NEG).topk(10, -1).indices.cpu()
            hit[name, l] = torch.tensor([bool(torch.isin(tf[p], torch.tensor([i for v in c.values() for i in v])).any())
                                         for p, c in enumerate(f.candidates)])
    best = {n: max(ctx.readout_layers, key=lambda l: float(hit[n, l].float().mean())) for n in lenses}
    print(f"[w3 qual] best FIT layer (filtered top-10 hit rate): "
          + ", ".join(f"{n} L{best[n]} {float(hit[n, best[n]].float().mean()):.3f}" for n in lenses), flush=True)
    for n in lenses:
        firsts = torch.stack([top_u[n, l][:, 0] for l in ctx.readout_layers]).flatten().tolist()
        share = sum(t in excl for t in firsts) / len(firsts)
        common = torch.bincount(torch.tensor(firsts)).topk(5)
        print(f"[w3 qual] {n}: share of unfiltered first guesses that are excluded tokens {share:.2f}; most common "
              f"first guesses {[repr(tok.decode([int(i)])) for i in common.indices]} x{common.values.tolist()}",
              flush=True)
    pick = []
    for task in ("association", "multihop", "multilingual", "poetry", "typo"):
        ps = [p for p, s in enumerate(f.eval_slugs) if s == task]
        pick += ps[:1] + ps[len(ps) // 2:len(ps) // 2 + 1]
    for p in pick:
        ids = {i for v in f.candidates[p].values() for i in v}
        print(f"\n[qual] {f.eval_slugs[p]} / {f.item_names[p]}  concept(s): {list(f.candidates[p])}", flush=True)
        for n in lenses:
            l = best[n]
            s = ", ".join(("*" if t in ids else "~" if t in excl else "") + repr(tok.decode([t]))[1:-1]
                          for t in top_u[n, l][p].tolist())
            print(f"   L{l:2d} {n:9s} {'hit' if hit[n, l][p] else '   '} {s}", flush=True)
    return lenses["JacA full"]
