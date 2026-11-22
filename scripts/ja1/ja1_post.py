"""ja1 post-processing (pod, jpp venv). Lenses are plain dicts {layer: [d, d] fp32} for auto/candidates/ja1_*.py.

  pooled  --dir CKPT --K k [--minus CKPT2] --out f.pt     pooled J = sum_p G_p / #p (half B: CKPT - CKPT2)
  a5      --dir CKPT [--minus CKPT2] --weights W.json --out f.pt   sum_e w_e J_e (stock expert means, min 50 positions)
  counts  --dir CKPT --K 8                                 per-layer expert position counts
  heldout --model M --out acts.pt                          200 held-out passages (raw text), last-token block outputs
  agree   --model M --a f1 --b f2 --acts acts.pt           per layer cos(flat), top-10 overlap on the 200 activations
  mass    --dir RUN                                        gradient-mass shares per layer (from RUN/mass.pt)
  lensfile --src jpp_lens.pt --model M --out f.pt          convert a stock lens file to a plain dict
"""
import argparse
import json
import os
import sys

sys.modules["fla"] = None
import torch as t  # noqa: E402

R = "/workspace/o-lens-improvement"


def load_sums(d, K):
    st = t.load(os.path.join(d, f"experts_K{K}_checkpoint.pt"), map_location="cpu", weights_only=True)
    return {int(l): (v["weighted_jacobian_row_sum_EFN"].float(), v["weight_sum_E"].float(), v["position_count_E"])
            for l, v in st["layer_sums"].items()}, st["config"]["num_prompts_trained_on"]


def get(a):
    """sum of the sums in --dir (several dirs: e.g. three wordings of the same passages) minus those in --minus."""
    s, n = None, 0
    for sign, dirs in ((1, a.dir), (-1, a.minus or [])):
        for d in dirs:
            x, m = load_sums(d, a.K)
            n += sign * m
            s = ({l: tuple(sign * v for v in x[l]) for l in x} if s is None
                 else {l: tuple(u + sign * v for u, v in zip(s[l], x[l])) for l in s})
    return s, n


def cmd_pooled(a):
    s, n = get(a)
    M = {l: r.sum(0) / w.sum() for l, (r, w, c) in s.items()}
    t.save(M, a.out)
    print(json.dumps(dict(out=a.out, prompts=n, positions=int(next(iter(s.values()))[2].sum()),
                          tr_over_d={l: float(t.trace(m) / m.shape[0]) for l, m in M.items()})))


def cmd_a5(a):
    a.K = 8
    s, n = get(a)
    W = json.load(open(a.weights))["by_layer"]
    M = {}
    for l, (r, w, c) in s.items():
        pooled = r.sum(0) / w.sum()
        ex = t.stack([pooled if (c[e] < 50 or w[e] <= 0) else r[e] / w[e] for e in range(r.shape[0])])
        M[l] = t.einsum("e,eij->ij", t.tensor(W[str(l)]["weights_unit_l2"]), ex)
    t.save(M, a.out)
    print(json.dumps(dict(out=a.out, prompts=n, counts={l: c.tolist() for l, (r, w, c) in s.items()})))


def cmd_counts(a):
    s, n = get(a)
    print(json.dumps(dict(prompts=n, counts={l: c.tolist() for l, (r, w, c) in s.items()})))


def cmd_heldout(a):
    sys.path.insert(0, "/root/ja1/code")
    from ja1_fit import LAYERS, load_model, passages
    from jlens.hooks import ActivationRecorder
    model = load_model(a.model)
    data, order = passages()
    layers = LAYERS[a.model]
    X = {l: [] for l in layers}
    with t.no_grad():
        for i in order[6000:6200]:  # never used for fits (<= 2048) or routers (< 1000)
            ids = model.tokenizer(data[i]["text"].strip("\n"), add_special_tokens=False, return_tensors="pt").input_ids
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(ids.to(model.input_device))
            for l in layers:
                X[l].append(rec.activations[l][0, -1].float().cpu())
    t.save({l: t.stack(v) for l, v in X.items()}, a.out)


def cmd_agree(a):
    sys.path.insert(0, f"{R}/scripts/perf1")
    from lite_model import get_lite_model
    model = get_lite_model(a.model)
    A, B = t.load(a.a, weights_only=True), t.load(a.b, weights_only=True)
    X = t.load(a.acts, weights_only=True)
    out = {}
    with t.no_grad():
        for l in sorted(A):
            ma, mb = A[l].cuda().float(), B[l].cuda().float()
            cos = float(t.nn.functional.cosine_similarity(ma.flatten(), mb.flatten(), dim=0))
            h = X[l].cuda().float()
            ta = model.unembed(h @ ma.T).float().topk(10).indices
            tb = model.unembed(h @ mb.T).float().topk(10).indices
            ov = float(sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)) / (10 * len(ta)))
            out[l] = dict(cos=round(cos, 4), top10=round(ov, 4))
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"))


def cmd_mass(a):
    a.dir = a.dir[0]
    m = t.load(os.path.join(a.dir, "mass.pt"), weights_only=False)
    res = {}
    for l in m["mass"][0]:
        last, psg, tot, first = [], [], [], []
        for mm, pos in zip(m["mass"], m["pos"]):
            v = mm[l]
            P = v[pos]
            last.append(float(P[-1] / P.sum()))
            first.append(float(P[0] / P.sum()))
            psg.append(float(P.sum() / v.sum()))
        res[l] = dict(last_share_of_passage=round(sum(last) / len(last), 4),
                      first_share_of_passage=round(sum(first) / len(first), 4),
                      uniform_share=round(sum(1 / len(p) for p in m["pos"]) / len(m["pos"]), 4),
                      passage_share_of_all_positions=round(sum(psg) / len(psg), 4))
    print(json.dumps(res))
    json.dump(res, open(os.path.join(a.dir, "mass_summary.json"), "w"), indent=1)


def cmd_lensfile(a):
    from workspace_lens.utils import load_lens_file
    L = load_lens_file(a.src, hf_model_name=a.model)
    t.save({int(l): v.float().cpu() for l, v in L.jacobians_L_dict_FN.items()}, a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--dir", nargs="+")
    ap.add_argument("--minus", nargs="*")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--weights")
    ap.add_argument("--out")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--acts")
    ap.add_argument("--src")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
