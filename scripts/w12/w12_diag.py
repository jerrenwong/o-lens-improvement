"""w12 diagnostics (pod, lite model).
junk  : DEV items, no filter: share of (item, readout layer) pairs whose top-1 token is junk (f4 token rule: no
        letter/digit, special/added, contains @ & < >), per lens (J, J++, each trained map alone), mean over layers and
        per layer -> runs/w12/junk_<tag>.json
qual  : 10 FIT-split prompts (every len(fit)//10-th fit item), top-10 (f4 mask, one token per word group) at one layer,
        J-lens vs trained map(s) -> runs/w12/qual_<tag>_L<layer>.txt
Usage: python w12_diag.py <model> junk cfg1,cfg2,..   |   python w12_diag.py <model> qual <layer> cfg1[,cfg2]"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
M, MODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts/perf1"))
from olens.readout import Bench  # noqa: E402
from lite_model import get_lite_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda"
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
JP = {"2b": ("artifacts/baselines/jlens_prefit.pt", "artifacts/baselines/jpp_prefit.pt"),
      "9b": ("runs/v6_9b/jlens9b.pt", "artifacts/baselines/jpp9b_prefit.pt")}[TAG]
bench = Bench.load(R / f"artifacts/bench/qwen3.5-{TAG}.pt")
split = json.loads((R / f"auto/split_{TAG}.json").read_text())
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
model = get_lite_model(M)
V = model._lm_head.weight.shape[0]
_fs = importlib.util.spec_from_file_location("w1_filters", R / "auto" / "filters.py")
FM = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(FM); sys.modules["w1_filters"] = FM
FL = FM.Filters(model.tokenizer, V, [bench])
F4 = FL.apply("f4")
strs = FL.texts
SPEC = FM.special_ids(model.tokenizer)
junk = torch.tensor([not (any(c.isalnum() for c in s) and i not in SPEC and not any(c in s for c in ("@", "&", "<", ">")))
                     for i, s in enumerate(strs)], device=dev)


def mats(name):
    if name in ("J", "JPP"):
        L = load_lens_file(str(R / JP[name == "JPP"]), hf_model_name=M)
        return {l: L.jacobians_L_dict_FN[l].to(dev).float() for l in LAYERS}
    m = torch.load(R / f".hf27/w12/{TAG}/maps/{name}.pt", map_location="cpu")
    return {int(l): v["M"].to(dev).float() for l, v in m.items()}


if MODE == "junk":
    idx = sorted(pos[tuple(k)] for k in split["dev"])
    names = ["J", "JPP"] + sys.argv[3].split(",")
    out = {}
    for nm in names:
        Mm = mats(nm)
        per = {}
        for l in LAYERS:
            h = bench.resid[l][torch.tensor(idx)].to(dev).float()
            top = model.unembed(h @ Mm[l].T).float().argmax(-1)
            per[l] = junk[top].float().mean().item()
        out[nm] = dict(mean=sum(per.values()) / len(per), per_layer=per)
        print(f"{nm:24s} junk top-1 share {out[nm]['mean']:.3f}  " + " ".join(f"L{l} {v:.2f}" for l, v in per.items()),
              flush=True)
    (R / f"runs/w12/junk_{TAG}.json").write_text(json.dumps(out, indent=1))
else:
    layer = int(sys.argv[3]); names = ["J"] + sys.argv[4].split(",")
    prompts = {}
    for f in glob.glob("/root/jpp_lens/data/jlens/evaluations/lens-eval-*.json"):
        for x in json.load(open(f))["items"]:
            prompts[(Path(f).stem.replace("lens-eval-", ""), x["name"])] = x["prompt"]
    fit = bench.index("fit")
    pick = fit[::max(1, len(fit) // 10)][:10]
    mask = torch.zeros(V, dtype=torch.bool, device=dev); mask[bench.excluded_ids.to(dev)] = True
    h = bench.resid[layer][torch.tensor(pick)].to(dev).float()
    tops = {}
    for nm in names:
        lg = model.unembed(h @ mats(nm)[layer].T).float().masked_fill(mask, float("-inf"))
        tops[nm] = [FM.group_topk(lg[j], F4.group_ids, 10) for j in range(len(pick))]
    lines = [f"{M} layer {layer}: top-10 (f4 mask, one token per word group), FIT-split prompts\n"]
    for j, i in enumerate(pick):
        key = (bench.eval_slugs[i], bench.item_names[i])
        lines.append(f"[{key[0]}/{key[1]}] prompt: {prompts.get(key, '?')!r}")
        lines.append(f"   answers: {list(bench.candidates[i].keys())}")
        for nm in names:
            lines.append(f"   {nm:14s}: " + ", ".join(repr(strs[t]) for t in tops[nm][j]))
        lines.append("")
    (R / f"runs/w12/qual_{TAG}_L{layer}.txt").write_text("\n".join(lines))
    print("\n".join(lines))
