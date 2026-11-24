"""w9 reader check (c), step 2: score lenses on DEV items with the residual taken at the readout token (offset 0) and
at 3 tokens earlier (offset 3), from .hf27/w9/early_<tag>.pt (w9_early.py). The ctx, lens build (incl. the lam search
on the FIT split) and scoring (readout_rows, pass_at_k) are those of auto/evaluate.py (imported, unchanged); only the
residuals handed to lens.transport differ. Offset 0 must reproduce the evaluator's dev score (checked).
Usage: python w9_early_score.py <model> <filter> <spec-file: name|env lines> -> runs/w9/early_<tag>_<filter>.json"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

R = Path("/workspace/o-lens-improvement")
M, FLT_LEVEL, SPEC = sys.argv[1:4]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts/perf1"))
_es = importlib.util.spec_from_file_location("w9_evaluate", R / "auto/evaluate.py")
EV = importlib.util.module_from_spec(_es); _es.loader.exec_module(EV)
from olens.readout import Bench, readout_rows  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402
from lite_model import get_lite_model  # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda"
cfg = EV.MODELS[M]
TAG = cfg["tag"]
bench = Bench.load(R / cfg["bench"])
split = json.loads((R / f"auto/split_{TAG}.json").read_text())
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
fit_idx = sorted(pos[tuple(k)] for k in split["fit"])
model = get_lite_model(M)


def lens_dict(path):
    L = load_lens_file(str(R / path), hf_model_name=M)
    return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}


_fs = importlib.util.spec_from_file_location("w1_filters", R / "auto" / "filters.py")
_fm = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(_fm); sys.modules["w1_filters"] = _fm
FL = _fm.Filters(model.tokenizer, model._lm_head.weight.shape[0], [bench])
FLT = FL.apply(FLT_LEVEL)
fi = torch.as_tensor(fit_idx, dtype=torch.long)
fit = SimpleNamespace(resid={l: v[fi].to(dev).float() for l, v in bench.resid.items()},
                      candidates=[bench.candidates[i] for i in fit_idx], eval_slugs=[bench.eval_slugs[i] for i in fit_idx],
                      item_names=[bench.item_names[i] for i in fit_idx], excluded_ids=bench.excluded_ids.clone(),
                      group_ids=FLT.group_ids, filter_level=FLT_LEVEL)
ctx = SimpleNamespace(model=model, J=lens_dict(cfg["J"]), JPP=lens_dict(cfg["JPP"]), fit=fit, dict_path=None,
                      wikitext=None, readout_layers=list(cfg["layers"]), device=dev, filters=FL)
E = torch.load(R / f".hf27/w9/early_{TAG}.pt")
idx = E["idx"]
assert idx == sorted(pos[tuple(k)] for k in split["dev"])
_cs = importlib.util.spec_from_file_location("w9_cand", R / "auto/candidates/w9_alens.py")
results = {}
outp = R / f"runs/w9/early_{TAG}_{FLT_LEVEL}.json"
if outp.exists():
    results = json.loads(outp.read_text())
for line in open(SPEC):
    if not line.strip() or line.startswith("#"):
        continue
    name, envs = line.strip().split("|")[:2]
    if name in results:
        continue
    for kv in envs.split():
        k, v = kv.split("=", 1); os.environ[k] = v
    cand = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(cand)
    lens = cand.build(ctx)
    res = {}
    for off in (0, 3):
        rows = []
        for layer in cfg["layers"]:
            h = E["off"][off][layer].to(dev).float()
            v = lens.transport(h, layer, None).float()
            rows += readout_rows(name, layer, bench, v, model, filtered=True, items=idx, chunk_size=1,
                                 group_ids=FLT.group_ids)
        p10, per = EV.pass_at_k(rows, 10); p1, _ = EV.pass_at_k(rows, 1)
        res[str(off)] = dict(p10=p10, p1=p1, per_task_p10=per)
    results[name] = res
    print(f"[w9 early] {name}: readout token {res['0']['p10']:.3f}/{res['0']['p1']:.3f}; 3 tokens earlier "
          f"{res['3']['p10']:.3f}/{res['3']['p1']:.3f}", flush=True)
    outp.write_text(json.dumps(results, indent=1))
