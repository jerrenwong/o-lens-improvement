"""w9 qualitative: 10 FIT-split benchmark prompts, top-10 tokens (f4 mask) of the A-lens alone with map pair A vs B at one
layer. Usage: python w9_qual.py <model> <layer> <stemA> <stemB>  (stem = e.g. lin_ce_98k -> E_<tag>_{topic,spec}_<stem>.pt)
Output: runs/w9/qual_<tag>_L<layer>_<stemA>_vs_<stemB>.txt"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
M, LAYER, SA, SB = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts/perf1")); sys.path.append(str(R / "auto/candidates"))
from olens.readout import Bench  # noqa: E402
from lite_model import get_lite_model  # noqa: E402
import w9_alens as W  # noqa: E402
from fm27_lib import unit  # noqa: E402

torch.set_grad_enabled(False)
TAG = "2b" if "2B" in M else "9b"
bench = Bench.load(R / f"artifacts/bench/qwen3.5-{TAG}.pt")
model = get_lite_model(M)
_fs = importlib.util.spec_from_file_location("w1_filters", R / "auto" / "filters.py")
_fm = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(_fm); sys.modules["w1_filters"] = _fm
FL = _fm.Filters(model.tokenizer, model._lm_head.weight.shape[0], [bench])
F = FL.apply("f4")
V = model._lm_head.weight.shape[0]
mask = torch.zeros(V, dtype=torch.bool, device="cuda"); mask[bench.excluded_ids.to("cuda")] = True
prompts = {}
for f in glob.glob("/root/jpp_lens/data/jlens/evaluations/lens-eval-*.json"):
    for x in json.load(open(f))["items"]:
        prompts[(Path(f).stem.replace("lens-eval-", ""), x["name"])] = x["prompt"]
fit = bench.index("fit")
pick = fit[::max(1, len(fit) // 10)][:10]
maps = {s: (W.load_maps_w9(str(R / f".hf27/w9/maps/E_{TAG}_topic_{s}.pt"), "cuda"),
            W.load_maps_w9(str(R / f".hf27/w9/maps/E_{TAG}_spec_{s}.pt"), "cuda")) for s in (SA, SB)}
lines = [f"{M} layer {LAYER}: A-lens alone, top-10 (f4 mask), {SA} vs {SB}; fit-split prompts\n"]
h = bench.resid[LAYER][torch.tensor(pick)].cuda().float()
tops = {}
for s, (E, E2) in maps.items():
    a = unit(unit(W.emap_w9(E, h, LAYER)) + unit(W.emap_w9(E2, h, LAYER)))
    lg = model.unembed(a).float().masked_fill(mask, float("-inf"))
    tops[s] = lg.topk(10, -1).indices.tolist()
for j, i in enumerate(pick):
    key = (bench.eval_slugs[i], bench.item_names[i])
    lines.append(f"[{key[0]}/{key[1]}] prompt: {prompts.get(key, '?')!r}")
    lines.append(f"   answers: {list(bench.candidates[i].keys())}")
    for s in (SA, SB):
        lines.append(f"   {s:14s}: " + ", ".join(repr(model.tokenizer.decode([t])) for t in tops[s][j]))
    lines.append("")
out = R / f"runs/w9/qual_{TAG}_L{LAYER}_{SA}_vs_{SB}.txt"
out.write_text("\n".join(lines))
print("\n".join(lines))
