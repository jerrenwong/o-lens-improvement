"""lr1: low rank on the J++ experts BEFORE combining (Qwen3.5-9B). Step-0 stats + all truncated files, one pass.

    python scripts/lr1_prep.py TAG EXPERTS.pt WEIGHTS.json JPP.pt OUTDIR V0=256,512 V1=256,512 V2=512 V3=16,64
V0 r: final combined matrix truncated to rank r                     -> OUTDIR/TAG_V0_r{r}.pt   (lens, fp32)
V1 r: each expert truncated to r, existing weights                  -> OUTDIR/TAG_V1_r{r}.pt   (lens)
V2 r: each expert truncated to r, experts file for fit-weights     -> OUTDIR/TAG_exV1_r{r}.pt (experts, fp16)
V3 r: Ebar (mean expert) full rank + D_k truncated to r             -> OUTDIR/TAG_V3a_r{r}.pt (lens, existing w)
                                                                       OUTDIR/TAG_exV3_r{r}.pt (experts, fp16)
Lenses combine the fp16-rounded truncated experts in fp32, as ExpertJacobians.combine does from a loaded file.
Step-0 stats -> OUTDIR/TAG_step0.json and stdout.
"""
import json
import sys
import time
from pathlib import Path

import torch as t
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.utils import load_lens_file

tag, ex_path, w_path, jpp_path, out = sys.argv[1:6]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
R = {k: [] for k in ("V0", "V1", "V2", "V3")}
for a in sys.argv[6:]:
    k, v = a.split("=")
    R[k] = [int(x) for x in v.split(",") if x]
ex_ranks = sorted(set(R["V2"]))  # expert files of V1-truncation needed for the refit
import os; dev = os.environ.get("LR1_DEV", "cuda"); t.set_num_threads(int(os.environ.get("LR1_THREADS", "64")))
t0 = time.time()
EJ = ExpertJacobians.load(ex_path)
W = json.load(open(w_path))["by_layer"]
Jref = load_lens_file(jpp_path, hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN
layers = EJ.layers
print(f"loaded {tag} in {time.time()-t0:.0f}s; layers {layers}; experts {EJ.num_experts}", flush=True)


def pr(S):  # participation ratio of squared singular values
    s2 = S.double() ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def trunc(U, S, Vh, r):
    return (U[:, :r] * S[:r]) @ Vh[:r]


lens_V0 = {r: {} for r in R["V0"]}
lens_V1 = {r: {} for r in R["V1"]}
lens_V3a = {r: {} for r in R["V3"]}
exV1 = {r: {} for r in ex_ranks}
exV3 = {r: {} for r in R["V3"]}
stats = {}
for l in layers:
    tl = time.time()
    E = EJ.experts_L_dict_EFN[l].to(dev, t.float32)
    w = t.tensor(W[str(l)]["weights_unit_l2"], dtype=t.float32, device=dev)
    J = t.einsum("e,eij->ij", w, E)
    jr = Jref[l].to(dev, t.float32)
    maxdiff = float((J - jr).abs().max()); rel = float((J - jr).norm() / jr.norm())
    Eb = E.mean(0); D = E - Eb
    sw = float(w.sum())
    A = sw * Eb; B = t.einsum("e,eij->ij", w, D)
    nJ2 = float(J.norm() ** 2)
    sE = [t.linalg.svd(E[k]) for k in range(E.shape[0])]
    sD = [t.linalg.svd(D[k]) for k in range(E.shape[0])]
    sJ = t.linalg.svd(J); sEb = t.linalg.svd(Eb)
    U0 = t.stack([s[0][:, 0] for s in sE])  # top output direction per expert
    cos = (U0 @ U0.T).abs()
    iu = t.triu_indices(8, 8, 1)
    pc = cos[iu[0], iu[1]]
    eb_cos = (U0 @ sEb[0][:, 0]).abs()
    st = dict(
        maxabs_rebuild_vs_file=maxdiff, rel_rebuild=rel,
        w=[round(x, 4) for x in w.tolist()], sum_w=sw,
        norm_Ebar=float(Eb.norm()), norm_D=[float(D[k].norm()) for k in range(8)],
        norm_E=[float(E[k].norm()) for k in range(8)], norm_J=nJ2 ** .5,
        share_mean=float(A.norm() ** 2) / nJ2, share_diff=float(B.norm() ** 2) / nJ2,
        share_cross=2 * float((A * B).sum()) / nJ2,
        pr_Ebar=pr(sEb[1]), pr_D=[pr(s[1]) for s in sD], pr_E=[pr(s[1]) for s in sE], pr_J=pr(sJ[1]),
        cos_top_out_pairs=dict(mean=float(pc.mean()), min=float(pc.min()), max=float(pc.max())),
        cos_top_out_expert_vs_Ebar=[round(x, 3) for x in eb_cos.tolist()],
        kept_E={r: float(sum((s[1][:r] ** 2).sum() / (s[1] ** 2).sum() for s in sE) / 8) for r in (16, 64, 256, 512, 1024, 2048)},
        kept_D={r: float(sum((s[1][:r] ** 2).sum() / (s[1] ** 2).sum() for s in sD) / 8) for r in (16, 64, 256, 1024)},
        kept_J={r: float((sJ[1][:r] ** 2).sum() / (sJ[1] ** 2).sum()) for r in (256, 512, 1024, 2048)},
    )
    stats[l] = st
    for r in R["V0"]:
        lens_V0[r][l] = trunc(*sJ, r).cpu()
    for r in sorted(set(R["V1"]) | set(ex_ranks)):
        Er = t.stack([trunc(*s, r) for s in sE]).half()
        if r in lens_V1:
            lens_V1[r][l] = t.einsum("e,eij->ij", w, Er.float()).cpu()
        if r in exV1:
            exV1[r][l] = Er.cpu()
    for r in R["V3"]:
        Er = t.stack([Eb + trunc(*s, r) for s in sD]).half()
        lens_V3a[r][l] = t.einsum("e,eij->ij", w, Er.float()).cpu()
        exV3[r][l] = Er.cpu()
    del E, D, sE, sD, sJ, sEb
    t.cuda.empty_cache()
    print(f"L{l} ({time.time()-tl:.0f}s) rebuild maxabs {maxdiff:.2e} rel {rel:.1e} | sum_w {sw:+.3f} | "
          f"|Ebar| {st['norm_Ebar']:.2f} |D_k| mean {sum(st['norm_D'])/8:.2f} |J| {st['norm_J']:.2f} | "
          f"share mean {st['share_mean']:.3f} diff {st['share_diff']:.3f} cross {st['share_cross']:+.3f} | "
          f"PR Ebar {st['pr_Ebar']:.1f} D med {sorted(st['pr_D'])[4]:.1f} E med {sorted(st['pr_E'])[4]:.1f} J {st['pr_J']:.1f} | "
          f"|cos u1| pairs mean {st['cos_top_out_pairs']['mean']:.2f}", flush=True)
(out / f"{tag}_step0.json").write_text(json.dumps(stats, indent=1))
cfg = EJ.config


def save_lens(Js, name):
    JacobianLens(jacobians=Js, config=cfg.as_jacobian(checkpoint_name=name)).save(str(out / f"{name}.pt"), dtype=t.float32)
    print("wrote", out / f"{name}.pt", flush=True)


for r, Js in lens_V0.items():
    save_lens(Js, f"{tag}_V0_r{r}")
for r, Js in lens_V1.items():
    save_lens(Js, f"{tag}_V1_r{r}")
for r, Js in lens_V3a.items():
    save_lens(Js, f"{tag}_V3a_r{r}")
for kind, d in (("exV1", exV1), ("exV3", exV3)):
    for r, Es in d.items():
        ej = ExpertJacobians(experts_L_dict_EFN=Es, pooled_jacobian_L_dict_FN=EJ.pooled_jacobian_L_dict_FN,
                             position_counts_L_dict_E=EJ.position_counts_L_dict_E,
                             weight_sums_L_dict_E=EJ.weight_sums_L_dict_E, fallback_L_dict_Bool_E=EJ.fallback_L_dict_Bool_E,
                             config=cfg, min_kept_positions=EJ.min_kept_positions)
        ej.save(str(out / f"{tag}_{kind}_r{r}.pt"))
        print("wrote", out / f"{tag}_{kind}_r{r}.pt", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
