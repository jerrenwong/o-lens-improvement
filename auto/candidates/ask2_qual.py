"""ask2 qualitative (FIT split only): for 3 fit items per task (association, multihop, multilingual, poetry), print
the concept(s) and top-8 tokens of J++, E_topic alone and E_spec alone at two mid layers. Returns plain J++
(so its dev score must reproduce the J++ baseline: a sanity check)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask2_lib import ELens, load_maps, spec_path, tag, topic_path  # noqa: E402
from olens.readout import unembed_logits  # noqa: E402

MID = {"9b": [12, 20], "27b": [24, 40]}


def build(ctx):
    mt = load_maps(topic_path(ctx), ctx.device)
    ms = load_maps(spec_path(ctx), ctx.device)
    tok = ctx.model.tokenizer
    f = ctx.fit
    pick = []
    for task in ("association", "multihop", "multilingual", "poetry"):
        pick += [p for p, s in enumerate(f.eval_slugs) if s == task][:3]
    lenses = {"J++": ELens(ctx.JPP, mt, "mix", 0.0), "E_topic": ELens(ctx.JPP, mt, "alone"),
              "E_spec": ELens(ctx.JPP, ms, "alone")}
    for p in pick:
        cands = f.candidates[p]
        ids = {i for v in cands.values() for i in v}
        print(f"\n[qual] {f.eval_slugs[p]} / {f.item_names[p]}  concept(s): {list(cands)}", flush=True)
        for l in MID[tag(ctx)]:
            h = f.resid[l][p:p + 1]
            for name, lens in lenses.items():
                lg = unembed_logits(ctx.model, lens.transport(h, l), f.excluded_ids)[0]
                top = lg.topk(8).indices.tolist()
                s = ", ".join(("*" if t in ids else "") + repr(tok.decode([t]).strip())[1:-1] for t in top)
                print(f"   L{l:2d} {name:8s} {s}", flush=True)
    return ELens(ctx.JPP, mt, "mix", 0.0)
