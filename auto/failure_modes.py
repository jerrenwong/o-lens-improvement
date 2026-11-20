"""How do J-lens and J++ fail when they miss? Top-10 slot classes vs the model's own continuation.

Uses ONLY fit + dev items of auto/split_9b.json (test items are dropped right after loading).
Run on the pod from /root/jpp_lens (eval data is read by relative path):

    cd /root/jpp_lens && .venv/bin/python /workspace/o-lens-improvement/auto/failure_modes.py
Writes auto/failure_modes.json and auto/failure_modes.md.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch as t
from jlens.hooks import ActivationRecorder
from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import _item_readout_and_candidates
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

from olens.readout import Bench, candidate_ranks

ROOT = Path("/workspace/o-lens-improvement")
MODEL = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
LENSES = {"J-lens": ROOT / "runs/v6_9b/jlens9b.pt", "J++": ROOT / "artifacts/baselines/jpp9b_prefit.pt"}
CLASSES = ["concept", "answer", "continuation", "prompt_echo", "non_latin", "other"]
LETTER = {"concept": "C", "answer": "A", "continuation": "N", "prompt_echo": "E", "non_latin": "X", "other": "O"}
BUCKETS = [(11, 30), (31, 100), (101, 1000), (1001, 10**9)]
BNAMES = ["11-30", "31-100", "101-1000", ">1000"]
EX_LAYERS = [12, 20, 28]


def bucket(r):
    for (lo, hi), nm in zip(BUCKETS, BNAMES):
        if lo <= r <= hi:
            return nm
    return None


def main():
    t.set_grad_enabled(False)
    split = json.loads((ROOT / "auto/split_9b.json").read_text())
    keep = {tuple(k): s for s in ("fit", "dev") for k in split[s]}
    test_keys = {tuple(k) for k in split["test"]}
    assert not (set(keep) & test_keys)

    bench = Bench.load(ROOT / "artifacts/bench/qwen3.5-9b.pt")
    idx = [p for p in range(bench.num_items) if (bench.eval_slugs[p], bench.item_names[p]) in keep]
    assert all((bench.eval_slugs[p], bench.item_names[p]) not in test_keys for p in idx)
    missing = set(keep) - {(bench.eval_slugs[p], bench.item_names[p]) for p in idx}
    print(f"using {len(idx)} items (fit+dev); {len(missing)} split keys not in bench", flush=True)

    model = get_hf_model(MODEL, attn_implementation="sdpa")
    tok = model.tokenizer
    last = model.n_layers - 1
    mats = {nm: load_lens_file(str(p), hf_model_name=MODEL).jacobians_L_dict_FN for nm, p in LENSES.items()}
    vecs = {}
    for ln, M in mats.items():
        for layer in LAYERS:
            Ml = M[layer].cuda().float()
            vecs[ln, layer] = (bench.resid[layer][idx].cuda().float() @ Ml.T).cpu()
            del Ml
    del mats
    t.cuda.empty_cache()
    excl = None
    raw = {(it.eval_slug, it.name): it for it in load_readout_eval_items()
           if (it.eval_slug, it.name) in keep}  # test items never kept

    dec_cache = {}

    def dec(i):
        if i not in dec_cache:
            dec_cache[i] = tok.decode([i])
        return dec_cache[i]

    items = []
    max_bench_err = 0.0
    for n_done, p in enumerate(idx):
        key = (bench.eval_slugs[p], bench.item_names[p])
        it = raw[key]
        rd = _item_readout_and_candidates(model, it, 512)
        pos = int(rd.readout_position)
        with ActivationRecorder(model.layers, at=[LAYERS[2], last]) as rec:
            model.forward(rd.input_ids_Int_1S)
        final = rec.activations[last][0, pos].float()
        chk = rec.activations[LAYERS[2]][0, pos].float().cpu()
        ref = bench.resid[LAYERS[2]][p]
        max_bench_err = max(max_bench_err, ((chk - ref).norm() / ref.norm()).item())
        lg = model.unembed(final[None]).float()[0]
        if excl is None:
            excl = t.zeros(lg.shape[-1], dtype=t.bool, device=lg.device)
            excl[bench.excluded_ids.to(lg.device)] = True
        lg = lg.masked_fill(excl, float("-inf"))
        cont50 = lg.topk(50).indices.tolist()
        cands = bench.candidates[p]
        cand_all = {i for v in cands.values() for i in v}
        tgt = (it.target or "").strip().lower()
        prompt_l = it.prompt.lower()

        def classify(i, cont_set):
            if i in cand_all:
                return "concept"
            s = dec(i).strip().lower()
            if tgt and s and (s == tgt or (len(s) >= 3 and tgt.startswith(s))):
                return "answer"
            if i in cont_set:
                return "continuation"
            if len(s) >= 3 and s in prompt_l:
                return "prompt_echo"
            if not any(c.isascii() and c.isalnum() for c in dec(i)):
                return "non_latin"
            return "other"

        c10, c50 = set(cont50[:10]), set(cont50)
        rec_item = {"eval": key[0], "name": key[1], "split": keep[key], "prompt": it.prompt[:200],
                    "target": it.target, "concepts": list(cands.keys()),
                    "cont_top5": [dec(i) for i in cont50[:5]], "lens": {}}
        for ln in LENSES:
            per = {}
            for layer in LAYERS:
                v = vecs[ln, layer][n_done].cuda()
                L = model.unembed(v[None]).float().masked_fill(excl[None], float("-inf"))
                rk = min(candidate_ranks(L, [cands])[0].values())
                top = L[0].topk(10).indices.tolist()
                per[layer] = {
                    "rank": int(rk), "top": [dec(i) for i in top],
                    "cls": [classify(i, c10) for i in top], "cls50": [classify(i, c50) for i in top],
                    "cos": t.nn.functional.cosine_similarity(v, final.to(v.device), dim=0).item(),
                    "ov10": len(set(top) & c10) / 10,
                }
            rec_item["lens"][ln] = per
        items.append(rec_item)
        if n_done % 25 == 0:
            print(f"{n_done}/{len(idx)}", flush=True)
    print(f"bench residual check at layer {LAYERS[2]}: max rel err {max_bench_err:.3g}", flush=True)
    t.cuda.empty_cache()
    aggregate(items, len(missing), max_bench_err)


def shares(cells, key="cls"):
    n = sum(len(c[key]) for c in cells)
    out = {k: 0.0 for k in CLASSES}
    for c in cells:
        for k in c[key]:
            out[k] += 1
    return {k: (v / n if n else None) for k, v in out.items()} | {"n_cells": len(cells)}


def top1(cells):
    out = {k: 0 for k in CLASSES}
    for c in cells:
        out[c["cls"][0]] += 1
    return {k: (v / len(cells) if cells else None) for k, v in out.items()} | {"n_cells": len(cells)}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def aggregate(items, n_missing, bench_err):
    tasks = sorted({it["eval"] for it in items})
    groups = {"all": items} | {tk: [it for it in items if it["eval"] == tk] for tk in tasks}
    res = {"n_items": len(items), "n_fit": sum(it["split"] == "fit" for it in items),
           "n_dev": sum(it["split"] == "dev" for it in items), "test_items_used": 0,
           "bench_resid_check_rel_err": bench_err, "split_keys_missing_from_bench": n_missing,
           "layers": LAYERS, "classes": CLASSES, "agg": {}}
    for ln in LENSES:
        res["agg"][ln] = {}
        for g, its in groups.items():
            d = {}
            for hm in ("all", "hit", "miss"):
                def cells(layer=None):
                    out = []
                    for it in its:
                        for l2 in LAYERS if layer is None else [layer]:
                            c = it["lens"][ln][l2]
                            if hm == "all" or (hm == "hit") == (c["rank"] <= 10):
                                out.append(c)
                    return out
                allc = cells()
                d[hm] = {
                    "slot_share": shares(allc), "slot_share_top50cont": shares(allc, "cls50"),
                    "top1_share": top1(allc), "cos": mean([c["cos"] for c in allc]),
                    "overlap10": mean([c["ov10"] for c in allc]),
                    "per_layer": {l: {"slot_share": shares(cells(l)), "slot_share_top50cont": shares(cells(l), "cls50"),
                                      "top1_share": top1(cells(l)),
                                      "cos": mean([c["cos"] for c in cells(l)]),
                                      "overlap10": mean([c["ov10"] for c in cells(l)])} for l in LAYERS},
                }
            # miss depth
            allmiss = [min(it["lens"][ln][l]["rank"] for l in LAYERS) for it in its
                       if all(it["lens"][ln][l]["rank"] > 10 for l in LAYERS)]
            cellmiss = [it["lens"][ln][l]["rank"] for it in its for l in LAYERS if it["lens"][ln][l]["rank"] > 10]
            d["n_items"] = len(its)
            d["n_hit_items"] = len(its) - len(allmiss)
            d["miss_depth_all_layers"] = {b: sum(bucket(r) == b for r in allmiss) for b in BNAMES} | {"n": len(allmiss)}
            d["miss_depth_cells"] = {b: sum(bucket(r) == b for r in cellmiss) for b in BNAMES} | {"n": len(cellmiss)}
            res["agg"][ln][g] = d
    # what J++ fixes
    fixed = [it for it in items if all(it["lens"]["J-lens"][l]["rank"] > 10 for l in LAYERS)
             and any(it["lens"]["J++"][l]["rank"] <= 10 for l in LAYERS)]
    fx = {"n_items": len(fixed), "by_task": {tk: sum(it["eval"] == tk for it in fixed) for tk in tasks}}
    for ln in LENSES:
        cs = [it["lens"][ln][min(LAYERS, key=lambda l: it["lens"]["J++"][l]["rank"])] for it in fixed]
        fx[ln] = shares(cs)
    res["jpp_fixes"] = fx
    # examples
    ex = {}
    for tk in tasks:
        miss = [it for it in groups[tk] if all(it["lens"]["J++"][l]["rank"] > 10 for l in LAYERS)][:6]
        ex[tk] = [{k: it[k] for k in ("name", "prompt", "concepts", "target", "cont_top5")}
                  | {"layers": {l: {"rank": it["lens"]["J++"][l]["rank"],
                                    "top": [f"{s!r}:{LETTER[c]}" for s, c in zip(it["lens"]["J++"][l]["top"], it["lens"]["J++"][l]["cls"])]}
                                for l in EX_LAYERS}} for it in miss]
    res["examples"] = ex
    out_json = ROOT / "auto/failure_modes.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    write_md(res, tasks)


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}"


def write_md(res, tasks):
    A = res["agg"]
    L = []
    w = L.append
    w("# Lens failure modes: Qwen3.5-9B, J-lens vs J++\n")
    w(f"Items: {res['n_items']} (fit {res['n_fit']} + dev {res['n_dev']}). Test items used: 0 (filtered at load).")
    w(f"Layers {LAYERS}. Top-10 with punctuation filter. Bench residual re-check rel err {res['bench_resid_check_rel_err']:.2g}.\n")
    w("SUMMARY_PLACEHOLDER\n")
    w("Classes (priority order): C concept, A answer (target), N continuation (model's filtered top-10 next tokens), "
      "E prompt echo, X non-latin, O other. Hit/miss is per (item, layer) cell.\n")

    w("## 1. Top-10 slot shares (%), all layers pooled\n")
    w("| lens | task | cells | " + " | ".join(CLASSES) + " | N@50 |")
    w("|---|---|---|" + "---|" * (len(CLASSES) + 1))
    for ln in LENSES:
        for g in ["all"] + tasks:
            for hm in ("hit", "miss"):
                d = A[ln][g][hm]
                s = d["slot_share"]
                w(f"| {ln} | {g} {hm} | {s['n_cells']} | " + " | ".join(pct(s[k]) for k in CLASSES)
                  + f" | {pct(d['slot_share_top50cont']['continuation'])} |")
    w("\nN@50: continuation share when the continuation set is the model's top-50.\n")

    w("## 2. Continuation share, cosine and overlap by layer (all tasks)\n")
    w("| lens | cells | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    w("|---|---|" + "---|" * len(LAYERS))
    for ln in LENSES:
        for hm in ("hit", "miss"):
            pl = A[ln]["all"][hm]["per_layer"]
            w(f"| {ln} | N% {hm} | " + " | ".join(pct(pl[l]["slot_share"]["continuation"]) for l in LAYERS) + " |")
        for hm in ("hit", "miss"):
            pl = A[ln]["all"][hm]["per_layer"]
            w(f"| {ln} | A% {hm} | " + " | ".join(pct(pl[l]["slot_share"]["answer"]) for l in LAYERS) + " |")
        pl = A[ln]["all"]["all"]["per_layer"]
        w(f"| {ln} | cos(Mh, final) | " + " | ".join(f"{pl[l]['cos']:.2f}" for l in LAYERS) + " |")
        w(f"| {ln} | overlap@10 | " + " | ".join(f"{pl[l]['overlap10']:.2f}" for l in LAYERS) + " |")
        w(f"| {ln} | miss cells | " + " | ".join(str(A[ln]['all']['miss']['per_layer'][l]['slot_share']['n_cells']) for l in LAYERS) + " |")
    w("")

    w("## 3. Top-1 token class (%), misses vs hits\n")
    w("| lens | task | " + " | ".join(CLASSES) + " |")
    w("|---|---|" + "---|" * len(CLASSES))
    for ln in LENSES:
        for g in ["all"] + tasks:
            for hm in ("hit", "miss"):
                s = A[ln][g][hm]["top1_share"]
                w(f"| {ln} | {g} {hm} | " + " | ".join(pct(s[k]) for k in CLASSES) + " |")
    w("")

    w("## 4. Cosine(Mh, final residual) and overlap@10, per task\n")
    w("| lens | task | cos hit | cos miss | ov@10 hit | ov@10 miss |")
    w("|---|---|---|---|---|---|")
    for ln in LENSES:
        for g in ["all"] + tasks:
            d = A[ln][g]
            f2 = lambda x: "-" if x is None else f"{x:.2f}"
            w(f"| {ln} | {g} | {f2(d['hit']['cos'])} | {f2(d['miss']['cos'])} | {f2(d['hit']['overlap10'])} | {f2(d['miss']['overlap10'])} |")
    w("")

    w("## 5. Miss depth (best rank over layers, items missing at all 7 layers)\n")
    w("| lens | task | items | hit items | all-miss | " + " | ".join(BNAMES) + " | cell misses " + " | ".join(BNAMES) + " |")
    w("|---|---|---|---|---|" + "---|" * 4 + "---|" + "---|" * 3)
    for ln in LENSES:
        for g in ["all"] + tasks:
            d = A[ln][g]
            m, c = d["miss_depth_all_layers"], d["miss_depth_cells"]
            w(f"| {ln} | {g} | {d['n_items']} | {d['n_hit_items']} | {m['n']} | " + " | ".join(str(m[b]) for b in BNAMES)
              + f" | {c['n']}: " + " | ".join(str(c[b]) for b in BNAMES) + " |")
    w("")

    fx = res["jpp_fixes"]
    w("## 6. What J++ fixes\n")
    w(f"Items where the J-lens misses at every layer and J++ hits: {fx['n_items']} ({fx['by_task']}).")
    w("Slot shares (%) at J++'s best layer:\n")
    w("| lens | " + " | ".join(CLASSES) + " |")
    w("|---|" + "---|" * len(CLASSES))
    for ln in LENSES:
        w(f"| {ln} | " + " | ".join(pct(fx[ln][k]) for k in CLASSES) + " |")
    w("")

    w("## 7. Examples: J++ misses at all layers\n")
    w("Tags: C concept, A answer, N continuation, E echo, X non-latin, O other.\n")
    for tk in tasks:
        w(f"### {tk}\n")
        if not res["examples"][tk]:
            w("(none)\n")
        for e in res["examples"][tk]:
            pr = e["prompt"].replace("\n", " ⏎ ").replace("|", "/")
            w(f"- **{e['name']}** concept `{', '.join(e['concepts'])}`, target `{e['target']}`, "
              f"model top-5 {e['cont_top5']}")
            w(f"  - prompt: {pr}")
            for l, d in e["layers"].items():
                w(f"  - L{l} (rank {d['rank']}): " + " ".join(d["top"]))
        w("")
    (ROOT / "auto/failure_modes.md").write_text("\n".join(L) + "\n")
    print("wrote report", flush=True)


if __name__ == "__main__":
    main()
