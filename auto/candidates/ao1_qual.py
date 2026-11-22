"""ao1 qualitative (FIT split only): 8 fit items (association, multihop, multilingual: one J++ hit + one J++ miss each;
poetry, typo: first item), top-8 filtered tokens of the ask-only lens (env AO1_J, AO1_E, AO1_LAM), J++, J_ask
alone and the map alone at two mid layers. '*' marks a candidate token. Returns the ask-only lens (its dev score
must reproduce the matching ao1 mix run: sanity)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import Lens, load_J, maps_E  # noqa: E402
from fm27_lib import excl_mask, logits, NEG  # noqa: E402

MID = {2048: [9, 15], 4096: [12, 20]}


def build(ctx):
    tok, f, mask = ctx.model.tokenizer, ctx.fit, excl_mask(ctx)
    J, E, lam = load_J(ctx), maps_E(ctx), float(os.environ["AO1_LAM"])
    lenses = {"ask-only": Lens([("J", J, 1.0), ("E", E, lam)]), "J++": Lens([("J", ctx.JPP, 1.0)]),
              "J_ask": Lens([("J", J, 1.0)]), "map": Lens([("E", E, 1.0)])}

    def hits(lens):
        hit = torch.zeros(len(f.eval_slugs), dtype=torch.bool)
        for l in ctx.readout_layers:
            top = logits(ctx.model, lens.transport(f.resid[l], l)).masked_fill(mask, NEG).topk(10, -1).indices.cpu()
            for p, c in enumerate(f.candidates):
                ids = torch.tensor([i for v in c.values() for i in v])
                hit[p] |= bool(torch.isin(top[p], ids).any())
        return hit

    hj, ha = hits(lenses["J++"]), hits(lenses["ask-only"])
    pick = []
    for task, n in (("association", 2), ("multihop", 2), ("multilingual", 2), ("poetry", 1), ("typo", 1)):
        ps = [p for p, s in enumerate(f.eval_slugs) if s == task]
        if n == 2:
            pick += [next((p for p in ps if hj[p]), None), next((p for p in ps if not hj[p]), None)]
        else:
            pick += ps[:1]
    pick = [p for p in pick if p is not None]
    print(f"[ao1 qual] lam {lam}; FIT any-layer top-10 hit counts (item has any candidate): J++ {int(hj.sum())}, "
          f"ask-only {int(ha.sum())} of {len(hj)}", flush=True)
    for p in pick:
        cands = f.candidates[p]
        ids = {i for v in cands.values() for i in v}
        print(f"\n[qual] {f.eval_slugs[p]} / {f.item_names[p]}  concept(s): {list(cands)}  "
              f"J++ {'hit' if hj[p] else 'miss'}, ask-only {'hit' if ha[p] else 'miss'}", flush=True)
        for l in MID[ctx.model.d_model]:
            h = f.resid[l][p:p + 1]
            for name, lens in lenses.items():
                lg = logits(ctx.model, lens.transport(h, l)).masked_fill(mask, NEG)[0]
                top = lg.topk(8).indices.tolist()
                s = ", ".join(("*" if t in ids else "") + repr(tok.decode([t]).strip())[1:-1] for t in top)
                print(f"   L{l:2d} {name:9s} {s}", flush=True)
    return lenses["ask-only"]
