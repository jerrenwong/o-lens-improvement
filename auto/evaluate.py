"""FIXED scorer for the lens autoresearch loop. Never edit (see auto/program.md).

    python auto/evaluate.py --lens auto/candidates/<name>.py [--split dev] [--model Qwen/Qwen3.5-9B]
    python auto/evaluate.py --lens auto/lens.py --split test --orchestrator

Loads the model once, builds ctx, times build(ctx), transports the split's benchmark rows at each
readout layer, ranks the vocabulary with the harness rule (olens.readout.readout_rows), writes
auto/runs/<name>/readout_{filtered,unfiltered}.csv and prints a summary block at the end.
Metric: item-weighted macro pass@k (per (item, intermediate) best rank over layers; per item the
share of its intermediates with rank <= k; mean per task; mean over tasks).
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
BUDGET_S = 600.0
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")

MODELS = {
    "Qwen/Qwen3.5-9B": dict(
        tag="9b", bench="artifacts/bench/qwen3.5-9b.pt", J="runs/v6_9b/jlens9b.pt",
        JPP="artifacts/baselines/jpp9b_prefit.pt", layers=[4, 8, 12, 16, 20, 24, 28],
        dicts=lambda layer, space: f"runs/bsf9b/L{layer}_{space}_b1_10M.pt",
    ),
    "Qwen/Qwen3.5-2B": dict(
        tag="2b", bench="artifacts/bench/qwen3.5-2b.pt", J="artifacts/baselines/jlens_prefit.pt",
        JPP="artifacts/baselines/jpp_prefit.pt", layers=[3, 6, 9, 12, 15, 18, 21],
        dicts=lambda layer, space: (f"runs/bsf1/L18_{space}_b1_10M.pt" if layer == 18
                                    else f"runs/bsf2/L{layer}_{space}_b1_10M.pt"),
    ),
}


def crash() -> None:
    traceback.print_exc()
    print("CRASH", flush=True)
    sys.exit(1)


def pass_at_k(rows: list[dict], k: int) -> tuple[float, dict[str, float]]:
    best: dict[tuple, int] = {}
    for r in rows:
        key = (r["eval"], r["item"], r["intermediate"])
        best[key] = min(best.get(key, r["rank"]), r["rank"])
    item_hits: dict[tuple, list[float]] = defaultdict(list)
    for (e, item, _), rank in best.items():
        item_hits[(e, item)].append(float(rank <= k))
    per_task: dict[str, list[float]] = defaultdict(list)
    for (e, _), hits in item_hits.items():
        per_task[e].append(sum(hits) / len(hits))
    per_task_mean = {e: sum(v) / len(v) for e, v in sorted(per_task.items())}
    return sum(per_task_mean.values()) / len(per_task_mean), per_task_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", choices=list(MODELS))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "heldout"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--orchestrator", action="store_true")
    ap.add_argument("--filter", default="f1", choices=["f1", "f2", "f3", "f4"],  # w1 (auto/filters.py)
                    help="token filter level of the filtered score; f1 = benchmark filter (default, unchanged)")
    ap.add_argument("--lite", action="store_true",
                    help="load only the final norm + LM head + tokenizer (scripts/perf1/lite_model.py); "
                         "candidates that need forward passes fail with a clear error")
    args = ap.parse_args()
    if args.split in ("test", "heldout") and not args.orchestrator:
        sys.exit(f"--split {args.split} is reserved for the orchestrator (pass --orchestrator)")
    lens_path = Path(args.lens).resolve()
    name = args.name or lens_path.stem
    cfg = MODELS[args.model]
    out_dir = ROOT / "auto" / "runs" / name
    t_start = time.time()

    sys.path.insert(0, str(ROOT / "src"))
    from workspace_lens import get_hf_model
    from workspace_lens.utils import load_lens_file

    from olens.readout import Bench, readout_rows, write_csv
    from olens.train import wikitext_windows

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    dev = "cuda"

    # ---- data and split
    bench = Bench.load(ROOT / cfg["bench"])
    split = json.loads((ROOT / f"auto/split_{cfg['tag']}.json").read_text())
    pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
    fit_idx = sorted(pos[tuple(k)] for k in split["fit"])
    assert fit_idx == bench.index("fit"), "split file's fit items differ from the bench's fit split"
    if args.split == "heldout":
        idx = bench.index("held-out")
    else:
        idx = sorted(pos[tuple(k)] for k in split[args.split])
    assert not set(idx) & set(fit_idx)

    # ---- model and ctx
    if args.lite:  # perf1: same unembed code and weights, no decoder layers loaded
        sys.path.insert(0, str(ROOT / "scripts" / "perf1"))
        from lite_model import get_lite_model, lite_blocked
        model = get_lite_model(args.model)
    else:
        model = get_hf_model(args.model, attn_implementation="sdpa")
        for p in model._hf_model.parameters():
            p.requires_grad_(False)

    def lens_dict(path):
        L = load_lens_file(str(ROOT / path), hf_model_name=args.model)
        return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}

    # w1: filter level (auto/filters.py). f1 leaves the benches untouched; f2/f3 replace their excluded_ids.
    _fs = importlib.util.spec_from_file_location("w1_filters", ROOT / "auto" / "filters.py")
    _fm = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(_fm); sys.modules["w1_filters"] = _fm
    FL = _fm.Filters(model.tokenizer, model._lm_head.weight.shape[0], [bench])
    FLT = FL.apply(args.filter)
    fi = torch.as_tensor(fit_idx, dtype=torch.long)
    fit = SimpleNamespace(
        resid={l: v[fi].to(dev).float() for l, v in bench.resid.items()},
        candidates=[bench.candidates[i] for i in fit_idx],
        eval_slugs=[bench.eval_slugs[i] for i in fit_idx],
        item_names=[bench.item_names[i] for i in fit_idx],
        excluded_ids=bench.excluded_ids.clone(),
        group_ids=FLT.group_ids, filter_level=args.filter,  # w1: f3 groups (None at f1/f2)
    )
    ctx = SimpleNamespace(
        model=model, J=lens_dict(cfg["J"]), JPP=lens_dict(cfg["JPP"]), fit=fit,
        dict_path=lambda layer, space: str(ROOT / cfg["dicts"](layer, space)),
        wikitext=(lite_blocked("ctx.wikitext") if args.lite else
                  lambda wsplit, n: wikitext_windows(model.tokenizer, wsplit, n, 128, Path("/root/cache"),
                                                     para_start=True)),
        readout_layers=list(cfg["layers"]), device=dev, filters=FL,  # w1
    )
    si = torch.as_tensor(idx, dtype=torch.long)
    rows_resid = {l: v[si].to(dev).float() for l, v in bench.resid.items()}

    # ---- candidate: import, build (timed), transport
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        spec = importlib.util.spec_from_file_location(f"candidate_{name}", lens_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        with torch.no_grad():
            lens = mod.build(ctx)
        torch.cuda.synchronize()
        build_s = time.time() - t0
        vecs = {}
        for layer in cfg["layers"]:
            h = rows_resid[layer].clone()
            resid_all = {l: v.clone() for l, v in rows_resid.items()}
            with torch.no_grad():
                v = lens.transport(h, layer, resid_all)
            if not isinstance(v, torch.Tensor) or tuple(v.shape) != tuple(h.shape):
                raise ValueError(f"transport at layer {layer} returned "
                                 f"{getattr(v, 'shape', type(v))}, expected {tuple(h.shape)}")
            v = v.detach().float()
            if not torch.isfinite(v).all():
                raise ValueError(f"transport at layer {layer} returned non-finite values")
            vecs[layer] = v
        torch.cuda.synchronize()
        peak_gb = (torch.cuda.max_memory_allocated() - base_mem) / 1e9
    except Exception:
        crash()

    # ---- score
    del lens
    rows = {True: [], False: []}
    for layer, v in vecs.items():
        for f in (True, False):
            rows[f] += readout_rows(name, layer, bench, v, model, filtered=f, items=idx, chunk_size=1, group_ids=FLT.group_ids)
    write_csv(out_dir / "readout_filtered.csv", rows[True])
    write_csv(out_dir / "readout_unfiltered.csv", rows[False])
    p10, per_task = pass_at_k(rows[True], 10)
    p1, _ = pass_at_k(rows[True], 1)
    p10u, _ = pass_at_k(rows[False], 10)
    p1u, _ = pass_at_k(rows[False], 1)
    total_s = time.time() - t_start
    (out_dir / "summary.json").write_text(json.dumps(dict(
        lens=str(lens_path), model=args.model, split=args.split, pass10=p10, pass1=p1, pass10_unf=p10u,
        pass1_unf=p1u, per_task_pass10=per_task, build_seconds=build_s, peak_vram_gb=peak_gb,
        n_items=len(idx), total_seconds=total_s, filter=args.filter, **({"lite": True} if args.lite else {})), indent=1))

    pre = args.split

    def line(label, value):
        print(label.ljust(17) + " " + value)

    if build_s > BUDGET_S:
        print("OVER_BUDGET")
    print("---")
    line("split:", args.split)
    line(f"{pre}_pass10:", f"{p10:.4f}")
    line(f"{pre}_pass1:", f"{p1:.4f}")
    line(f"{pre}_pass10_unf:", f"{p10u:.4f}")
    line(f"{pre}_pass1_unf:", f"{p1u:.4f}")
    line("per_task_pass10:", " ".join(f"{t} {per_task.get(t, float('nan')):.2f}" for t in TASKS))
    line("build_seconds:", f"{build_s:.1f}")
    line("peak_vram_gb:", f"{peak_gb:.1f}")
    print(f"{'n_items:'.ljust(17)} {len(idx)}", flush=True)


if __name__ == "__main__":
    main()
