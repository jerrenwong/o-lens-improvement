"""tl1 copy of auto/evaluate_x_v4.py (PREREG_v4 Addendum F, text-only control). Changes vs auto/evaluate_x_v4.py
(nothing about scoring): this docstring line, --verify-fwd, and the T-lens input feature: ctx.fit.tfeat [n_fit, d]
for the FIT rows (original bench), and transport(h, layer, resid_all, tfeat=...) with the v4 rows' features (prompts
from the md5-checked data/evals_v4 files) for candidates whose transport has a `tfeat` parameter.

xm1 copy of auto/evaluate_v4.py for the non-Qwen models of PREREG_v5. Changes vs auto/evaluate_v4.py (nothing
about scoring): this docstring line, the three xm1 model entries in MODELS (with bench_v4), --lite loads
scripts/xm1/lite_model_x.py, and --dry-run (allowed without --orchestrator): everything up to and including build + transport of the v4 rows (shape
and finiteness checks), then exit WITHOUT ranking, writing or printing any score (for checking the pipeline).

FIXED scorer for the lens autoresearch loop: v4 copy of auto/evaluate_v3.py (third fresh prompt set).

v4: `--split test` scores the frozen v4 prompts (data/evals_v4, bench artifacts/bench/v4_*.pt, split
auto/split_v4_<tag>.json). ctx.fit is built exactly as in evaluate27.py from the ORIGINAL bench and the
ORIGINAL fit list. Only Qwen3.5-9B and Qwen3.6-27B have v4 caches.

    python auto/evaluate_v4.py --lens auto/candidates/<name>.py --model Qwen/Qwen3.5-9B --split test --orchestrator

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
import inspect
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
        tag="9b", bench="artifacts/bench/qwen3.5-9b.pt", bench_v4="artifacts/bench/v4_qwen3.5-9b.pt", J="runs/v6_9b/jlens9b.pt",
        JPP="artifacts/baselines/jpp9b_prefit.pt", layers=[4, 8, 12, 16, 20, 24, 28],
        dicts=lambda layer, space: f"runs/bsf9b/L{layer}_{space}_b1_10M.pt",
    ),
    "Qwen/Qwen3.5-2B": dict(
        tag="2b", bench="artifacts/bench/qwen3.5-2b.pt", bench_v4="artifacts/bench/v4_qwen3.5-2b.pt", J="artifacts/baselines/jlens_prefit.pt",
        JPP="artifacts/baselines/jpp_prefit.pt", layers=[3, 6, 9, 12, 15, 18, 21],
        dicts=lambda layer, space: (f"runs/bsf1/L18_{space}_b1_10M.pt" if layer == 18
                                    else f"runs/bsf2/L{layer}_{space}_b1_10M.pt"),
    ),
    "Qwen/Qwen3.6-27B": dict(
        tag="27b", bench="artifacts/bench/qwen3.6-27b.pt", bench_v4="artifacts/bench/v4_qwen3.6-27b.pt", J="artifacts/baselines/jlens27b_prefit.pt",
        JPP="artifacts/baselines/jpp27b_released.pt", layers=[8, 16, 24, 32, 40, 48, 56],
        dicts=lambda layer, space: f"runs/bsf27b/L{layer}_{space}_b1_10M.pt",  # none exist for 27B
    ),
    "allenai/Olmo-3-7B-Instruct": dict(  # xm1
        tag="olmo7b", bench=".hf27/xm1/bench_olmo7b.pt", bench_v4=".hf27/xm1/v4_bench_olmo7b.pt", J="runs/xm1/olmo7b/pooled_jlens.pt",
        JPP="runs/xm1/olmo7b/jpp_olmo7b.pt", layers=[4, 8, 12, 16, 20, 24, 28],
        dicts=lambda layer, space: f"runs/bsfx/L{layer}_{space}_b1_10M.pt",  # none exist for xm1 models
    ),
    "google/gemma-4-12B-it": dict(  # xm1
        tag="gemma12b", bench=".hf27/xm1/bench_gemma12b.pt", bench_v4=".hf27/xm1/v4_bench_gemma12b.pt", J="runs/xm1/gemma12b/pooled_jlens.pt",
        JPP="runs/xm1/gemma12b/jpp_gemma12b.pt", layers=[6, 12, 18, 24, 30, 36, 42],
        dicts=lambda layer, space: f"runs/bsfx/L{layer}_{space}_b1_10M.pt",  # none exist for xm1 models
    ),
    "google/gemma-4-E4B-it": dict(  # xm1 reserve
        tag="gemma4e4b", bench=".hf27/xm1/bench_gemma4e4b.pt", bench_v4=".hf27/xm1/v4_bench_gemma4e4b.pt", J="runs/xm1/gemma4e4b/pooled_jlens.pt",
        JPP="runs/xm1/gemma4e4b/jpp_gemma4e4b.pt", layers=[5, 10, 15, 21, 26, 31, 36],
        dicts=lambda layer, space: f"runs/bsfx/L{layer}_{space}_b1_10M.pt",  # none exist for xm1 models
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
    ap.add_argument("--split", default="test", choices=["test"])  # v4: test = all included v4 items
    ap.add_argument("--name", default=None)
    ap.add_argument("--orchestrator", action="store_true")
    ap.add_argument("--lite", action="store_true",
                    help="load only the final norm + LM head + tokenizer (scripts/perf1/lite_model.py); "
                         "candidates that need forward passes fail with a clear error")
    ap.add_argument("--dry-run", action="store_true",
                    help="xm1: build + transport the v4 rows, then exit without ranking or scoring")
    ap.add_argument("--verify-fwd", action="store_true",
                    help="tl1: full model only; check per row that the token ids / readout position reproduce the "
                         "bench's cached residuals (as scripts/why1_v3.py)")
    args = ap.parse_args()
    if args.split in ("test", "heldout") and not args.orchestrator and not args.dry_run:  # xm1: dry run scores nothing
        sys.exit(f"--split {args.split} is reserved for the orchestrator (pass --orchestrator)")
    lens_path = Path(args.lens).resolve()
    name = args.name or lens_path.stem
    cfg = MODELS[args.model]
    if "bench_v4" not in cfg:
        sys.exit(f"no v4 cache for {args.model}")
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
    # v4: scored rows come from the v4 bench; the fit rows above stay on the original bench.
    split_v4 = json.loads((ROOT / f"auto/split_v4_{cfg['tag']}.json").read_text())
    assert split_v4["fit"] == split["fit"], "v4 split's fit list differs from the original fit list"
    bench_v4 = Bench.load(ROOT / cfg["bench_v4"])
    assert bench_v4.model == bench.model == args.model and bench_v4.layers == bench.layers
    pos_v4 = {k: i for i, k in enumerate(zip(bench_v4.eval_slugs, bench_v4.item_names, strict=True))}
    idx = sorted(pos_v4[tuple(k)] for k in split_v4[args.split])

    # ---- model and ctx
    if args.lite:  # perf1: same unembed code and weights, no decoder layers loaded
        sys.path.insert(0, str(ROOT / "scripts" / "xm1"))  # xm1
        from lite_model_x import get_lite_model, lite_blocked
        model = get_lite_model(args.model)
    else:
        model = get_hf_model(args.model, attn_implementation="sdpa")
        for p in model._hf_model.parameters():
            p.requires_grad_(False)

    def lens_dict(path):
        L = load_lens_file(str(ROOT / path), hf_model_name=args.model)
        return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}

    fi = torch.as_tensor(fit_idx, dtype=torch.long)
    fit = SimpleNamespace(
        resid={l: v[fi].to(dev).float() for l, v in bench.resid.items()},
        candidates=[bench.candidates[i] for i in fit_idx],
        eval_slugs=[bench.eval_slugs[i] for i in fit_idx],
        item_names=[bench.item_names[i] for i in fit_idx],
        excluded_ids=bench.excluded_ids.clone(),
    )
    ctx = SimpleNamespace(
        model=model, J=lens_dict(cfg["J"]), JPP=lens_dict(cfg["JPP"]), fit=fit,
        dict_path=lambda layer, space: str(ROOT / cfg["dicts"](layer, space)),
        wikitext=(lite_blocked("ctx.wikitext") if args.lite else
                  lambda wsplit, n: wikitext_windows(model.tokenizer, wsplit, n, 128, Path("/root/cache"),
                                                     para_start=True)),
        readout_layers=list(cfg["layers"]), device=dev,
    )
    si = torch.as_tensor(idx, dtype=torch.long)
    rows_resid = {l: v[si].to(dev).float() for l, v in bench_v4.resid.items()}

    # ---- tl1: T-lens input feature of every FIT row and every scored row: mean INPUT embedding of the row's prompt
    # tokens [0 .. readout position] (scripts/tl1/tl1_lib.py; positions asserted equal to the bench cache)
    sys.path.insert(0, str(ROOT / "scripts" / "tl1"))
    import tl1_lib
    EMB = tl1_lib.load_embedding(args.model)
    fit_seqs = tl1_lib.row_tokens(model.tokenizer, bench, fit_idx, tl1_lib.load_items(v4=False))
    row_seqs = tl1_lib.row_tokens(model.tokenizer, bench_v4, idx, tl1_lib.load_items(v4=True))
    if args.verify_fwd:  # tl1: as scripts/why1_v3.py, full model forward per row vs the cached residuals
        emb_hf = model._hf_model.get_input_embeddings().weight
        assert emb_hf.shape == EMB.shape and torch.equal(emb_hf.detach().cpu(), EMB.to(emb_hf.dtype)), \
            "safetensors embedding differs from the loaded model's input embedding"
        tl1_lib.verify_forward(model, bench, fit_idx, fit_seqs)
        tl1_lib.verify_forward(model, bench_v4, idx, row_seqs)
    fit.tfeat = tl1_lib.mean_embed(EMB, fit_seqs, dev)
    rows_tfeat = tl1_lib.mean_embed(EMB, row_seqs, dev)
    del EMB
    print(f"[tl1] features: fit {tuple(fit.tfeat.shape)}, scored rows {tuple(rows_tfeat.shape)}", flush=True)

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
        takes_tfeat = "tfeat" in inspect.signature(lens.transport).parameters  # tl1
        vecs = {}
        for layer in cfg["layers"]:
            h = rows_resid[layer].clone()
            resid_all = {l: v.clone() for l, v in rows_resid.items()}
            with torch.no_grad():
                v = (lens.transport(h, layer, resid_all, tfeat=rows_tfeat.clone()) if takes_tfeat  # tl1
                     else lens.transport(h, layer, resid_all))
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

    if args.dry_run:  # xm1: no ranks, no files, no scores
        print(f"DRY_RUN_OK model {args.model} lens {name}: built in {build_s:.1f}s, transported {len(idx)} v4 rows "
              f"at layers {list(vecs)}; nothing ranked or written", flush=True)
        return

    # ---- score
    del lens
    rows = {True: [], False: []}
    for layer, v in vecs.items():
        for f in (True, False):
            rows[f] += readout_rows(name, layer, bench_v4, v, model, filtered=f, items=idx, chunk_size=1)
    write_csv(out_dir / "readout_filtered.csv", rows[True])
    write_csv(out_dir / "readout_unfiltered.csv", rows[False])
    p10, per_task = pass_at_k(rows[True], 10)
    p1, _ = pass_at_k(rows[True], 1)
    p10u, _ = pass_at_k(rows[False], 10)
    p1u, _ = pass_at_k(rows[False], 1)
    total_s = time.time() - t_start
    (out_dir / "summary.json").write_text(json.dumps(dict(
        lens=str(lens_path), model=args.model, split="v4-" + args.split, pass10=p10, pass1=p1, pass10_unf=p10u,
        pass1_unf=p1u, per_task_pass10=per_task, build_seconds=build_s, peak_vram_gb=peak_gb,
        n_items=len(idx), total_seconds=total_s, **({"lite": True} if args.lite else {})), indent=1))

    pre = "v4_" + args.split

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
