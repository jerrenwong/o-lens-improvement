"""Export per-item, per-layer top-10 lens readouts for the qualitative viewer (site/readouts.html).

Lenses: J-lens, J++, and their indirect-only versions (M - I). For every benchmark item (fit and
held-out) at its readout position: the top-10 tokens (unfiltered, and with Readout Filtering) and
the rank of each hidden intermediate. Run from /root/jpp_lens (eval data is read by relative path).

    cd /root/jpp_lens && python .../export_readouts.py --model M --bench B --jlens J.pt --jpp P.pt \
        --layers 3,6,... --out readouts_2b.json
"""

import argparse
import json
from pathlib import Path

import torch as t
from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import _item_readout_and_candidates
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

from olens.readout import Bench, candidate_ranks

TOPK = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--jpp", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t.set_grad_enabled(False)
    model = get_hf_model(args.model, attn_implementation="sdpa")
    tok = model.tokenizer
    bench = Bench.load(args.bench)
    layers = [int(x) for x in args.layers.split(",")]
    J = load_lens_file(args.jlens, hf_model_name=args.model).jacobians_L_dict_FN
    P = load_lens_file(args.jpp, hf_model_name=args.model).jacobians_L_dict_FN
    lens_names = ["J-lens", "J++", "J − I (indirect only)", "J++ − I"]
    n = len(bench.item_names)

    vocab: dict[int, int] = {}
    strings: list[str] = []

    def vid(token_id: int) -> int:
        if token_id not in vocab:
            vocab[token_id] = len(strings)
            strings.append(tok.decode([token_id]))
        return vocab[token_id]

    excl = None
    # top[mode][item][layer][lens] -> [10 vocab idx]; rank[mode][item][layer][lens] -> [rank per intermediate]
    top = {m: [[[None] * 4 for _ in layers] for _ in range(n)] for m in "fu"}
    rank = {m: [[[None] * 4 for _ in layers] for _ in range(n)] for m in "fu"}
    inter_names = [list(c.keys()) for c in bench.candidates]
    for li, layer in enumerate(layers):
        h = bench.resid[layer].cuda().float()
        Jl, Pl = J[layer].cuda().float(), P[layer].cuda().float()
        vecs = [h @ Jl.T, h @ Pl.T, h @ Jl.T - h, h @ Pl.T - h]
        for xi, v in enumerate(vecs):
            for s in range(0, n, 64):
                lg = model.unembed(v[s : s + 64]).float()
                if excl is None:
                    excl = t.zeros(lg.shape[-1], dtype=t.bool, device=lg.device)
                    excl[bench.excluded_ids.to(lg.device)] = True
                cands = bench.candidates[s : s + 64]
                for mode, L in (("u", lg), ("f", lg.masked_fill(excl, float("-inf")))):
                    ids = L.topk(TOPK, dim=-1).indices.tolist()
                    rk = candidate_ranks(L, cands)
                    for r in range(len(ids)):
                        top[mode][s + r][li][xi] = [vid(i) for i in ids[r]]
                        rank[mode][s + r][li][xi] = [rk[r][nm] for nm in inter_names[s + r]]
        print(f"layer {layer} done", flush=True)

    raw = {(it.eval_slug, it.name): it for it in load_readout_eval_items()}
    items = []
    for p in range(n):
        it = raw[(bench.eval_slugs[p], bench.item_names[p])]
        rd = _item_readout_and_candidates(model, it, 512)
        ids = rd.input_ids_Int_1S[0].tolist()
        items.append({
            "e": bench.eval_slugs[p], "n": bench.item_names[p], "s": bench.split[p], "target": it.target,
            "inter": inter_names[p],
            "cand": [[vid(i) for i in bench.candidates[p][nm]] for nm in inter_names[p]],
            "toks": [vid(i) for i in ids], "pos": int(rd.readout_position),
            "top": {m: top[m][p] for m in "fu"}, "rank": {m: rank[m][p] for m in "fu"},
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "layers": layers, "lenses": lens_names, "vocab": strings, "items": items},
        ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {args.out}: {n} items, {len(strings)} distinct tokens, {Path(args.out).stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
