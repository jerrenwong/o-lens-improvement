"""W2b step 4 (diagnostic, not a benchmark): names / numbers.
Templated prompts whose next token is a number word, day, month, capital or surname. Kept only if the target is one
token and the MODEL's own top-1 next token is the target. At the last prompt token the block outputs at the readout
layers are read with J-lens and J++ (v = h J_l^T, z = unembed(v)), with alpha = 0 and with the alpha chosen on the
FIT split by eval_copred.py for that lens (per cut variant). Measure, among the target's family (other numbers /
days / months / capitals / surnames of the same template family): is the target ranked above every other member?
Per lens x variant: #prompts won at >= 1 layer, total (prompt, layer) wins, mean within-family rank (best layer).
Usage: python numnames.py <model> <eval json for J> <eval json for JPP>   (jsons from runs/w2/copred_eval/)"""
import json
import sys
from pathlib import Path

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/w2")
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

torch.set_grad_enabled(False)
M, JJ, JPPJ = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = sys.argv[:1]
import eval_copred as EC  # noqa: E402

cfg = EC.MODELS[M]
LAYERS = cfg["layers"]
dev = "cuda"
NUM = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen " \
      "eighteen nineteen twenty".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
CAP = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "Russia": "Moscow",
       "Egypt": "Cairo", "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna", "Poland": "Warsaw",
       "Ireland": "Dublin", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen",
       "Hungary": "Budapest", "Kenya": "Nairobi", "Peru": "Lima", "Cuba": "Havana", "Thailand": "Bangkok",
       "China": "Beijing", "England": "London", "Belgium": "Brussels", "Turkey": "Ankara", "Iran": "Tehran",
       "Argentina": "Buenos", "Canada": "Ottawa", "Australia": "Canberra", "Ukraine": "Kyiv", "Chile": "Santiago",
       "Colombia": "Bogota", "Iraq": "Baghdad", "Syria": "Damascus", "Lebanon": "Beirut", "Vietnam": "Hanoi"}
NAMES = [("The physicist Albert", "Einstein"), ("The painter Pablo", "Picasso"), ("The composer Wolfgang Amadeus", "Mozart"),
         ("The playwright William", "Shakespeare"), ("The scientist Isaac", "Newton"), ("The naturalist Charles", "Darwin"),
         ("The inventor Thomas", "Edison"), ("The painter Vincent van", "Gogh"), ("The composer Ludwig van", "Beethoven"),
         ("The philosopher Immanuel", "Kant"), ("The physicist Niels", "Bohr"), ("The chemist Marie", "Curie"),
         ("The writer Mark", "Twain"), ("The poet Emily", "Dickinson"), ("The writer Leo", "Tolstoy"),
         ("The scientist Galileo", "Galilei"), ("The painter Claude", "Monet"), ("The writer Jane", "Austen"),
         ("The composer Johann Sebastian", "Bach"), ("The economist Adam", "Smith"), ("The philosopher Karl", "Marx"),
         ("The president Abraham", "Lincoln"), ("The leader Mahatma", "Gandhi"), ("The physicist Stephen", "Hawking"),
         ("The writer Ernest", "Hemingway"), ("The painter Leonardo da", "Vinci"), ("The explorer Christopher", "Columbus"),
         ("The psychologist Sigmund", "Freud"), ("The writer Charles", "Dickens"), ("The poet Edgar Allan", "Poe")]


def prompts():
    P = []
    for a in range(1, 10):
        for b in range(1, 10):
            if a + b <= 20 and (a * 7 + b * 3) % 2 == 0:
                P.append(("add", f"{NUM[a].capitalize()} plus {NUM[b]} equals", NUM[a + b], NUM))
    for s in range(1, 17):
        P.append(("count", f"{NUM[s].capitalize()}, {NUM[s + 1]}, {NUM[s + 2]},", NUM[s + 3], NUM))
    for i, d in enumerate(DAYS):
        P.append(("day", f"The day after {d} is", DAYS[(i + 1) % 7], DAYS))
        P.append(("day", f"The day before {d} is", DAYS[(i - 1) % 7], DAYS))
        P.append(("day", f"{d}, {DAYS[(i + 1) % 7]},", DAYS[(i + 2) % 7], DAYS))
    for i, m in enumerate(MONTHS):
        P.append(("month", f"The month after {m} is", MONTHS[(i + 1) % 12], MONTHS))
        P.append(("month", f"The month before {m} is", MONTHS[(i - 1) % 12], MONTHS))
        P.append(("month", f"{m}, {MONTHS[(i + 1) % 12]},", MONTHS[(i + 2) % 12], MONTHS))
    for c, cap in CAP.items():
        P.append(("capital", f"The capital of {c} is", cap, list(CAP.values())))
    for p, n in NAMES:
        P.append(("name", p, n, [x for _, x in NAMES]))
    return P


model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
V = model._lm_head.weight.shape[0]


def one(w):
    t = tok(" " + w, add_special_tokens=False).input_ids
    return t[0] if len(t) == 1 else None


rows = []
for fam, text, tgt, members in prompts():
    t = one(tgt)
    fam_ids = sorted({x for x in (one(m) for m in members) if x is not None} - {t})
    if t is None or len(fam_ids) < 2:
        continue
    ids = torch.tensor([tok(text).input_ids], device=dev)
    last = len(model.layers) - 1
    with ActivationRecorder(model.layers, at=sorted(set(LAYERS) | {last})) as rec:
        model.forward(ids)
        acts = {l: rec.activations[l][0, -1].float() for l in LAYERS}
        top1 = model.unembed(rec.activations[last][:, -1]).float()[0].argmax().item()
    rows.append(dict(fam=fam, text=text, tgt=t, fam_ids=fam_ids, ok=top1 == t, acts=acts, top1=tok.decode([top1])))
n_all = len(rows)
rows = [r for r in rows if r["ok"]]
print(f"[numnames] {n_all} single-token prompts, {len(rows)} answered correctly by the model; per family: "
      + ", ".join(f"{f} {sum(r['fam'] == f for r in rows)}" for f in ("add", "count", "day", "month", "capital", "name")),
      flush=True)
torch.save([{k: v for k, v in r.items()} for r in rows], f"{R}/.hf27/w2/numnames_{cfg['tag']}.pt")
lenses = {n: {int(l): v.to(dev).float() for l, v in load_lens_file(f"{R}/{cfg[n]}", hf_model_name=M).jacobians_L_dict_FN.items()}
          for n in ("J", "JPP")}
chosen = {"J": json.load(open(JJ))["chosen"], "JPP": json.load(open(JPPJ))["chosen"]}
Qs = {}
out = {}
for ln, J in lenses.items():
    Z = {l: torch.stack([model.unembed((r["acts"][l] @ J[l].T)[None]).float()[0] for r in rows]) for l in LAYERS}
    settings = [("alpha0", None, 0.0)] + [(f"{v}@{c['alpha']}", v, c["alpha"]) for v, c in chosen[ln].items()
                                          if not v.endswith("_c")]
    settings += [(f"{v}@1.0", v, 1.0) for v in chosen[ln] if not v.endswith("_c")]
    for name, var, a in settings:
        if var is not None and var not in Qs:
            Qs[var] = EC.load_Q(f"{R}/.hf27/w2/copred_{cfg['tag']}_{var}.pt", V, dev)
        wins = torch.zeros(len(rows), len(LAYERS), dtype=torch.bool)
        frank = torch.zeros(len(rows), len(LAYERS))
        for j, l in enumerate(LAYERS):
            z = Z[l]
            if a > 0:
                Q, _ = Qs[var]
                z = z - a * torch.sparse.mm(Q, z.T.contiguous()).T
            for i, r in enumerate(rows):
                zt, zf = z[i, r["tgt"]], z[i, r["fam_ids"]]
                wins[i, j] = bool((zt > zf).all())
                frank[i, j] = 1 + (zf > zt).sum().item()
        best = frank.min(1).values
        per_fam = {f: int(sum(wins[i].any() for i, r in enumerate(rows) if r["fam"] == f))
                   for f in ("add", "count", "day", "month", "capital", "name")}
        out[f"{ln}/{name}"] = dict(won_any_layer=int(wins.any(1).sum()), wins_total=int(wins.sum()),
                                   cells=wins.numel(), mean_best_family_rank=best.mean().item(), per_family=per_fam,
                                   per_layer=wins.sum(0).tolist())
        o = out[f"{ln}/{name}"]
        print(f"{ln:4s} {name:14s} won>=1 layer {o['won_any_layer']:3d}/{len(rows)}  cell wins {o['wins_total']:4d}/"
              f"{o['cells']}  mean best family rank {o['mean_best_family_rank']:.2f}  per layer {o['per_layer']}  "
              f"per family {per_fam}", flush=True)
json.dump(dict(n_prompts=n_all, n_correct=len(rows), results=out), open(f"{R}/runs/w2/numnames_{cfg['tag']}.json", "w"),
          indent=1)
