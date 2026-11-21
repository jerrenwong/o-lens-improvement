"""ex1: what do the 8 J++ experts read, and what do the negative-weight experts subtract? (FIT split only)

    /root/jpp_lens/.venv/bin/python scripts/ex1/ex1_run.py --tag 9b
    /root/jpp_lens/.venv/bin/python scripts/ex1/ex1_run.py --tag 2b

Inputs (read-only): the run's router_K8.pt, experts_K8.pt, *_expert_weights.json, combined lens, pooled lens.
Text windows: prompts 500..699 of runs/r1_2b/prompts_raw.json (the pool the experts were fitted on; the
fits used prompts 0..127), tokenized like the fit (max_seq_len 128, positions 16..S-2).
Readouts: the fit:0 items, obtained exactly as jpp_cli fit-weights does (load_items_and_exclusions with
items_spec="fit:0"), readout position and candidates via the readout-eval helper.
Writes runs/ex1/<tag>/raw_<tag>.json (all per-item top-10 lists etc.) and clusters_<tag>.json.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

R = "/workspace/o-lens-improvement"
JPP = "/root/jpp_lens"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", choices=["9b", "2b"], required=True)
ap.add_argument("--n-windows", type=int, default=200)
ap.add_argument("--win-start", type=int, default=500)
ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
args = ap.parse_args()

CFG = {
    "9b": dict(model="Qwen/Qwen3.5-9B", layers=[4, 8, 12, 16, 20, 24, 28], router=f"{R}/runs/r2_9b/A/router_K8.pt",
               run=f"{R}/runs/r2_9b/A/A_64a", experts="experts_K8.pt", weights="jpp_A_64a_expert_weights.json",
               full="jpp_A_64a.pt", pooled="pooled_A_64a.pt", stamp="jpp_A_64a_stamp.json",
               bench=f"{R}/artifacts/bench/qwen3.5-9b.pt"),
    "2b": dict(model="Qwen/Qwen3.5-2B", layers=[3, 6, 9, 12, 15, 18, 21], router=f"{R}/runs/r1_2b/A/router_K8.pt",
               run=f"{R}/runs/r1_2b/A", experts="experts_K8.pt", weights="jpp_A_expert_weights.json",
               full="jpp_A.pt", pooled="pooled_A.pt", stamp="jpp_A_stamp.json",
               bench=f"{R}/artifacts/bench/qwen3.5-2b.pt"),
}[args.tag]
if args.device == "cpu":
    sys.modules["fla"] = None  # fla's triton kernels need CUDA; transformers then uses its torch GatedDeltaNet path
OUT = f"{R}/runs/ex1/{args.tag}"
os.makedirs(OUT, exist_ok=True)

os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
sys.path.insert(0, f"{R}/src")
import torch as t  # noqa: E402

import jpp_cli  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402
from workspace_lens.utils import get_position_mask_with_early_skips, load_lens_file, record_activations  # noqa: E402

from olens.readout import candidate_ranks  # noqa: E402

t.set_grad_enabled(False)
dev = args.device
if dev == "cpu":
    t.set_num_threads(24)
LAYERS = CFG["layers"]
model = get_hf_model(CFG["model"], attn_implementation="sdpa", device_map=("cpu" if dev == "cpu" else None))
tok = model.tokenizer
router = ActivationRouterCollection.load(CFG["router"])
print("[ex1] model + router loaded", flush=True)

# ---------------- part 1b: cluster membership on text windows ----------------
prompts = json.load(open(f"{R}/runs/r1_2b/prompts_raw.json"))[args.win_start: args.win_start + args.n_windows]
tok_counts = {L: [Counter() for _ in range(8)] for L in LAYERS}
pos_sum = {L: [0.0] * 8 for L in LAYERS}
all_counts = Counter()
n_pos = 0
for pi, p in enumerate(prompts):
    ids, acts = record_activations(model, p, 128, LAYERS)
    S = ids.shape[1]
    try:
        m = get_position_mask_with_early_skips(S, 16)
    except ValueError:
        continue
    pos = t.where(m)[0]
    tid = ids[0, pos.to(ids.device)].tolist()
    all_counts.update(tid)
    n_pos += len(tid)
    for L in LAYERS:
        a = router.assign(acts[L][pos.to(acts[L].device)].float(), L).cpu().tolist()
        for x, c, ps in zip(tid, a, pos.tolist(), strict=True):
            tok_counts[L][c][x] += 1
            pos_sum[L][c] += ps
print(f"[ex1] windows done: {n_pos} positions", flush=True)
vocab_str = {}


def s(i):
    i = int(i)
    if i not in vocab_str:
        vocab_str[i] = tok.decode([i])
    return vocab_str[i]


clusters = dict(n_windows=len(prompts), n_positions=n_pos, all_counts={s(k): v for k, v in all_counts.items()},
                layers={})
for L in LAYERS:
    clusters["layers"][str(L)] = [dict(n=sum(tok_counts[L][c].values()),
                                       mean_pos=pos_sum[L][c] / max(1, sum(tok_counts[L][c].values())),
                                       counts={str(k): v for k, v in tok_counts[L][c].items()})
                                  for c in range(8)]
clusters["all_counts_ids"] = {str(k): v for k, v in all_counts.items()}

# ---------------- fit items ----------------
_, fit_items, excluded = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=READOUT_EVALS_DATA_DIR, correctness_csv=f"{R}/data/model_correctness.csv",
    hf_model_name=CFG["model"], layers=LAYERS, readout_max_seq_len=512, items_spec="fit:0", fit_fraction=0.2)
stamp = json.load(open(f"{CFG['run']}/{CFG['stamp']}"))
excl = t.as_tensor(sorted(int(i) for i in excluded), dtype=t.long)
excl_set = set(excl.tolist())
items, H = [], {L: [] for L in LAYERS}
for it in fit_items:
    ro = re_mod._item_readout_and_candidates(model, it, 512)
    if ro is None:
        continue
    ids = ro.input_ids_Int_1S
    rp = int(ro.readout_position)
    _, acts = record_activations(model, it.prompt, 512, LAYERS)
    for L in LAYERS:
        H[L].append(acts[L][rp].float())
    logits = model._hf_model(input_ids=ids).logits[0, rp].float()
    top5 = logits.topk(5).indices.tolist()
    items.append(dict(eval=it.eval_slug, name=it.name, prompt=it.prompt, pos=rp, n_tok=int(ids.shape[1]),
                      prompt_ids=ids[0].tolist(), next_id=top5[0], next5=top5,
                      cand={k: [int(x) for x in v] for k, v in ro.candidate_ids_by_intermediate.items()}))
H = {L: t.stack(v).to(dev) for L, v in H.items()}
names = [it["name"] for it in items]
print(f"[ex1] {len(items)} fit items; stamp lists {len(stamp.get('fitting_item_names', []))}; "
      f"same set: {set(names) == set(stamp.get('fitting_item_names', names))}", flush=True)

# sanity: compare with bench fit rows (fit rows only)
try:
    from olens.readout import Bench
    b = Bench.load(CFG["bench"])
    fi = b.index("fit")
    key = {(b.eval_slugs[i], b.item_names[i]): i for i in fi}
    cs = []
    for k, it in enumerate(items):
        j = key.get((it["eval"], it["name"]))
        if j is not None:
            cs.append(float(t.nn.functional.cosine_similarity(H[LAYERS[3]][k], b.resid[LAYERS[3]][j].to(dev), dim=0)))
    print(f"[ex1] bench check: {len(cs)} matched, min cos {min(cs):.5f}", flush=True)
    del b
except Exception as e:  # noqa: BLE001
    print("[ex1] bench check skipped:", repr(e), flush=True)

cands = [it["cand"] for it in items]
next_ids = t.tensor([it["next_id"] for it in items], device=dev)

# ---------------- lenses ----------------
EJ = ExpertJacobians.load(f"{CFG['run']}/{CFG['experts']}")
W = json.load(open(f"{CFG['run']}/{CFG['weights']}"))["by_layer"]
full_file = load_lens_file(f"{CFG['run']}/{CFG['full']}", hf_model_name=CFG["model"]).jacobians_L_dict_FN
pooled_file = load_lens_file(f"{CFG['run']}/{CFG['pooled']}", hf_model_name=CFG["model"]).jacobians_L_dict_FN
tasks = sorted({it["eval"] for it in items})


def readout(V):
    """top-10 filtered / unfiltered ids, candidate ranks filtered / unfiltered."""
    out = {}
    for mode in ("f", "u"):
        lg = model.unembed(V).float()
        if mode == "f":
            lg[:, excl.to(lg.device)] = float("-inf")
        top = lg.topk(10, dim=-1).indices.cpu().tolist()
        rk = candidate_ranks(lg, cands)
        out[mode] = dict(top=top, rank=rk)
        for row in top:
            for i in row:
                s(i)
        del lg
    return out


def pass10(rk):
    per = defaultdict(list)
    for it, r in zip(items, rk, strict=True):
        per[it["eval"]].append(sum(v <= 10 for v in r.values()) / len(r))
    pt = {e: sum(v) / len(v) for e, v in per.items()}
    return sum(pt.values()) / len(pt), pt


raw = dict(tag=args.tag, model=CFG["model"], layers=LAYERS, items=items, excluded_n=len(excl_set), layers_out={})
for L in LAYERS:
    E = EJ.experts_L_dict_EFN[L].to(dev, t.float32)  # [8, F, N]
    w = t.tensor(W[str(L)]["weights_unit_l2"], device=dev)
    shares = W[str(L)]["signed_shares"]
    Pm = t.einsum("e,eij->ij", w.clamp(min=0), E)
    Nm = t.einsum("e,eij->ij", (-w).clamp(min=0), E)
    Fm = full_file[L].to(dev, t.float32)
    Pool = pooled_file[L].to(dev, t.float32)
    comb_err = float((Pm - Nm - Fm).norm() / Fm.norm())

    def fro_cos(A, B):
        return float((A * B).sum() / (A.norm() * B.norm()))

    h = H[L]
    assign = router.assign(h, L).cpu().tolist()
    readers = {f"e{e}": h @ E[e].T for e in range(8)}
    alpha = float((Nm * Pool).sum() / (Pool * Pool).sum())  # N's least-squares projection on the pooled lens
    readers.update(P=h @ Pm.T, N=h @ Nm.T, full=h @ Fm.T, pooled=h @ Pool.T, logit=h.clone(),
                   PmQ=h @ (Pm - alpha * Pool).T, Nres=h @ (Nm - alpha * Pool).T)
    res = {}
    for nm, V in readers.items():
        r = readout(V)
        p10f, ptf = pass10(r["f"]["rank"])
        p10u, ptu = pass10(r["u"]["rank"])
        top1u = t.tensor([row[0] for row in r["u"]["top"]], device=dev)
        r.update(pass10_f=p10f, pass10_f_task=ptf, pass10_u=p10u, pass10_u_task=ptu,
                 next_agree_u=float((top1u == next_ids).float().mean()),
                 next_in_top10_u=sum(it["next_id"] in row for it, row in zip(items, r["u"]["top"], strict=True)) / len(items),
                 norm=V.norm(dim=-1).cpu().tolist())
        res[nm] = r
    Ph, Nh, Fh, Qh = readers["P"], readers["N"], readers["full"], readers["pooled"]
    cosf = t.nn.functional.cosine_similarity
    geo = dict(cos_PN=cosf(Ph, Nh, dim=-1).cpu().tolist(), cos_N_pooled=cosf(Nh, Qh, dim=-1).cpu().tolist(),
               cos_P_pooled=cosf(Ph, Qh, dim=-1).cpu().tolist(), cos_full_pooled=cosf(Fh, Qh, dim=-1).cpu().tolist(),
               cos_N_h=cosf(Nh, h, dim=-1).cpu().tolist(), cos_P_h=cosf(Ph, h, dim=-1).cpu().tolist(),
               fro_cos_P_N=fro_cos(Pm, Nm), fro_cos_N_pooled=fro_cos(Nm, Pool), fro_cos_P_pooled=fro_cos(Pm, Pool),
               fro_cos_full_pooled=fro_cos(Fm, Pool), alpha_N_on_pooled=alpha,
               fro_cos_full_PmQ=fro_cos(Fm, Pm - alpha * Pool),
               fro_norm=dict(P=float(Pm.norm()), N=float(Nm.norm()), full=float(Fm.norm()), pooled=float(Pool.norm())),
               expert_fro_cos=[[fro_cos(E[a], E[b]) for b in range(8)] for a in range(8)],
               expert_fro_cos_pooled=[fro_cos(E[a], Pool) for a in range(8)],
               expert_fro_norm=[float(E[a].norm()) for a in range(8)])
    raw["layers_out"][str(L)] = dict(weights=W[str(L)]["weights_unit_l2"], shares=shares, comb_rel_err=comb_err,
                                     fit_counts=EJ.position_counts_L_dict_E[L].tolist(), assign=assign,
                                     readers=res, geo=geo)
    print(f"[ex1] L{L}: comb_err {comb_err:.2e}  full p10 {res['full']['pass10_f']:.3f}  "
          f"P {res['P']['pass10_f']:.3f}  N {res['N']['pass10_f']:.3f}", flush=True)
    del E, Pm, Nm, Fm, Pool, readers
    if dev == "cuda":
        t.cuda.empty_cache()

for it in items:
    for i in it["prompt_ids"] + it["next5"] + [x for v in it["cand"].values() for x in v]:
        s(i)
raw["vocab"] = {str(k): v for k, v in vocab_str.items()}
raw["excluded_in_vocab"] = [k for k in vocab_str if k in excl_set]
clusters["vocab"] = {str(k): s(k) for k in all_counts}
clusters["excluded_ids_in_windows"] = [k for k in all_counts if k in excl_set]
with open(f"{OUT}/clusters_{args.tag}.json", "w") as f:
    json.dump(clusters, f, ensure_ascii=False)
with open(f"{OUT}/raw_{args.tag}.json", "w") as f:
    json.dump(raw, f, ensure_ascii=False)
print(f"[ex1] wrote {OUT}", flush=True)
