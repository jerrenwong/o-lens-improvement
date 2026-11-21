"""perf1: bitwise check that get_lite_model(name) == get_hf_model(name) for everything --lite exposes.

Loads the full model (get_hf_model, as auto/evaluate*.py do) and the lite model in one process and
compares: norm/head weights (torch.equal), unembed outputs (torch.equal) on random vectors and on the
FIT-split bench residuals transported by J++ at every readout layer (row-at-a-time like the scorer and
in chunks of 64), the closed-form final_linear (W, gamma), the tokenizer (BOS flag, vocab, encodings),
d_model/n_layers/_logit_softcap, and that blocked attributes raise. Uses only the fit split.

    python scripts/perf1/check_lite.py --model Qwen/Qwen3.5-9B --out scripts/perf1/results/check_lite_9b.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "perf1"))
sys.path.insert(0, str(ROOT / "auto"))

BENCH = {"Qwen/Qwen3.5-9B": ("artifacts/bench/qwen3.5-9b.pt", "artifacts/baselines/jpp9b_prefit.pt"),
         "Qwen/Qwen3.6-27B": ("artifacts/bench/qwen3.6-27b.pt", "artifacts/baselines/jpp27b_released.pt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BENCH))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    from lite_model import LiteModelError, get_lite_model
    from olens.readout import Bench, final_linear
    from workspace_lens import get_hf_model
    from workspace_lens.utils import load_lens_file

    res = {"model": args.model}
    t0 = time.time()
    lite = get_lite_model(args.model)
    torch.cuda.synchronize()
    res["lite_load_s"] = time.time() - t0
    res["lite_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
    t0 = time.time()
    full = get_hf_model(args.model, attn_implementation="sdpa")
    torch.cuda.synchronize()
    res["full_load_s"] = time.time() - t0

    res["norm_class_same"] = type(full._final_norm) is type(lite._final_norm)
    res["head_class_same"] = type(full._lm_head) is type(lite._lm_head)
    res["norm_weight_equal"] = bool(torch.equal(full._final_norm.weight, lite._final_norm.weight))
    res["head_weight_equal"] = bool(torch.equal(full._lm_head.weight, lite._lm_head.weight))
    res["dtypes"] = [str(full._lm_head.weight.dtype), str(lite._lm_head.weight.dtype),
                     str(full._final_norm.weight.dtype), str(lite._final_norm.weight.dtype)]
    res["norm_eps"] = [getattr(full._final_norm, "eps", None), getattr(lite._final_norm, "eps", None)]
    res["attrs_equal"] = dict(d_model=full.d_model == lite.d_model, n_layers=full.n_layers == lite.n_layers,
                              softcap=full._logit_softcap == lite._logit_softcap)

    bench = Bench.load(ROOT / BENCH[args.model][0])
    fit_idx = bench.index("fit")
    L = load_lens_file(str(ROOT / BENCH[args.model][1]), hf_model_name=args.model)
    M = {int(k): v.cuda().float() for k, v in L.jacobians_L_dict_FN.items()}
    fi = torch.as_tensor(fit_idx)
    g = torch.Generator().manual_seed(0)
    probes = {"random": torch.randn(256, full.d_model, generator=g).cuda() * 3 + 0.5}
    for layer in bench.layers:
        if layer in M:
            probes[f"jpp_fit_L{layer}"] = bench.resid[layer][fi].cuda().float() @ M[layer].T
    n_rows, n_bad1, n_bad64 = 0, 0, 0
    for name, v in probes.items():
        for i in range(v.shape[0]):  # scorer call shape (chunk_size=1)
            a = full.unembed(v[i:i + 1]).float()
            b = lite.unembed(v[i:i + 1]).float()
            n_bad1 += int(not torch.equal(a, b))
        for s in range(0, v.shape[0], 64):
            n_bad64 += int(not torch.equal(full.unembed(v[s:s + 64]).float(), lite.unembed(v[s:s + 64]).float()))
        n_rows += v.shape[0]
    res["unembed_rows_checked"] = n_rows
    res["unembed_rows_not_bitwise_equal_chunk1"] = n_bad1
    res["unembed_chunks_not_bitwise_equal_chunk64"] = n_bad64

    Wf, gf = final_linear(full)
    Wl, gl = final_linear(lite)
    res["final_linear_equal"] = bool(torch.equal(Wf, Wl) and torch.equal(gf, gl))
    del Wf, Wl

    tf, tl = full.tokenizer, lite.tokenizer
    texts = ["The capital of France is", "  multi  space\nnewline", "Größe 東京 😀", "a" * 300, ""]
    res["tokenizer"] = dict(
        cls_same=type(tf) is type(tl), add_bos=[getattr(tf, "add_bos_token", None), getattr(tl, "add_bos_token", None)],
        len_equal=len(tf) == len(tl), vocab_equal=tf.get_vocab() == tl.get_vocab(),
        encodings_equal=all(tf(t).input_ids == tl(t).input_ids for t in texts),
        batch_equal=tf(texts, padding=True).input_ids == tl(texts, padding=True).input_ids,
    )
    blocked = {}
    for attr in ("layers", "forward", "_hf_model", "input_device", "encode"):
        try:
            getattr(lite, attr)
            blocked[attr] = "NOT BLOCKED"
        except LiteModelError:
            blocked[attr] = "raises LiteModelError"
    res["blocked"] = blocked
    res["full_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["ALL_OK"] = bool(res["norm_class_same"] and res["head_class_same"] and res["norm_weight_equal"]
                         and res["head_weight_equal"] and all(res["attrs_equal"].values())
                         and n_bad1 == 0 and n_bad64 == 0 and res["final_linear_equal"]
                         and all(v for k, v in res["tokenizer"].items() if k != "add_bos")
                         and res["tokenizer"]["add_bos"][0] == res["tokenizer"]["add_bos"][1]
                         and all(v.startswith("raises") for v in blocked.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
