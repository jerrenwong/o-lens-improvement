"""W2a diagnostic: how often the UNFILTERED top-1 token of a map read alone is a non-word (special/added token,
whitespace-only, or no letter/digit), on the DEV bench rows at the readout layers (mean over layers), and the same on
the passage TEST prefixes 1000..1999 (fit scripts' held-out passages). Maps: ridge (old) t3 / s5, CE maps per K,
and the A-lens average unit(unit(E h) + unit(E2 h)).
Usage: python nonword.py <model> [K ...]"""
import json
import re
import sys
from pathlib import Path

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402
from olens.readout import Bench  # noqa: E402

torch.set_grad_enabled(False)
M = sys.argv[1]
KS = sys.argv[2:] or ["10", "50", "200"]
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
BENCH = {"9b": "artifacts/bench/qwen3.5-9b.pt", "2b": "artifacts/bench/qwen3.5-2b.pt"}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "2b": f"{R}/.hf27/pod_root/e1/acts_2b.pt"}[TAG]
dev = "cuda"
model = get_lite_model(M)
tok = model.tokenizer
V = model._lm_head.weight.shape[0]
strs = [tok.decode([i]) for i in range(V)]
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
nonword = torch.tensor([not re.search(r"[^\W_]", s) or s.strip() == "" for s in strs], device=dev)
nonword[[i for i in special if i < V]] = True


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def load(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


def emap(m, h, l):
    mx, W, my = m[l]
    return (h - mx) @ W + my


bench = Bench.load(f"{R}/{BENCH}")
split = json.loads(Path(f"{R}/auto/split_{TAG}.json").read_text())
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
di = torch.tensor(sorted(pos[tuple(k)] for k in split["dev"]))
Hdev = {l: bench.resid[l][di].to(dev).float() for l in LAYERS}
Xs = torch.load(XFILE, map_location="cpu")["X"]
Hte = {l: Xs[l][1000:2000].to(dev).float() for l in LAYERS}
del Xs
pairs = {"ridge": (f"{R}/.hf27/ask3/E_{TAG}_t3.pt", f"{R}/.hf27/ask5/E_{TAG}_s5.pt")}
for k in KS:
    pairs[f"ce_k{k}"] = (f"{R}/.hf27/w2/E_{TAG}_topic_ce_k{k}.pt", f"{R}/.hf27/w2/E_{TAG}_spec_ce_k{k}.pt")
for var in ("shuf", "const"):
    for k in KS:
        p = (f"{R}/.hf27/w2/E_{TAG}_topic_{var}_k{k}.pt", f"{R}/.hf27/w2/E_{TAG}_spec_{var}_k{k}.pt")
        if Path(p[0]).exists() and Path(p[1]).exists():
            pairs[f"{var}_k{k}"] = p
out = {}
print(f"{'maps':12s} {'set':5s} | topic  spec   A-lens   (share of unfiltered top-1 that is a non-word; mean over layers)")
for name, (pt, ps) in pairs.items():
    Et, Es = load(pt), load(ps)
    for sname, H in (("dev", Hdev), ("pass", Hte)):
        r = {"topic": [], "spec": [], "alens": []}
        ex = []
        for l in LAYERS:
            a, b = unit(emap(Et, H[l], l)), unit(emap(Es, H[l], l))
            for key, v in (("topic", a), ("spec", b), ("alens", unit(a + b))):
                t1 = model.unembed(v).float().argmax(-1)
                r[key].append(nonword[t1].float().mean().item())
                if key == "alens" and sname == "dev":
                    ex += [strs[i] for i in t1[nonword[t1]].tolist()]
        m = {k: sum(v) / len(v) for k, v in r.items()}
        out[f"{name}/{sname}"] = dict(mean=m, per_layer=r)
        from collections import Counter
        print(f"{name:12s} {sname:5s} | {m['topic']:.3f}  {m['spec']:.3f}  {m['alens']:.3f}"
              + (f"   A-lens non-word top-1s: {Counter(ex).most_common(6)}" if sname == "dev" else ""), flush=True)
json.dump(out, open(f"{R}/runs/w2/nonword_{TAG}.json", "w"), indent=1)
