"""mlp1 step 0: refit ridge M3 (model A) per layer exactly as mp1_fit.py (M_all, last token, (eps, c) on val words,
all styles), reproduce test M alone / Jgen / Jrel / mix(Jgen + lam M, lam on val any-of-7). Saves the chosen ridge
per layer (.hf27/mlp1/ridge_L{l}.pt: mx, P, Wr, my, cfg) used to initialise B / C, and base ranks
(.hf27/mlp1/base_ranks.pt). Usage: mlp1_sanity.py [layers comma-separated]"""
import json, sys
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import *  # noqa

torch.set_grad_enabled(False)
layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else READ
gpu_setup()
D = Data()
JG, JR = lensd(JGEN), lensd(JREL)
log(f"train {len(D.TRN)} val {len(D.VAL)} test {len(D.TE)}; vocab {D.V}")
VR, TR, SR = defaultdict(dict), defaultdict(dict), defaultdict(dict)
choice = {}
for l in layers:
    X = D.load_X(l)
    xv, xt, xs = D.rows(X, D.VAL), D.rows(X, D.TE), D.rows(X, D.TRS)
    best = None
    for cfg, m in fit_M3(D, X, D.TRN):
        r = D.ranks_of(apply_M(m, xv), D.wi[D.VAL])
        z = (round(t10(r), 6), t1(r))
        if best is None or z > best[0]:
            best = (z, cfg, {k: v.clone() for k, v in m.items()})
    z, cfg, m = best
    choice[l] = dict(cfg=cfg, val_top10=z[0])
    torch.save(dict({k: v.cpu() for k, v in m.items()}, cfg=cfg), f"{OUT}/ridge_L{l}.pt")
    mv, mt = apply_M(m, xv), apply_M(m, xt)
    VR["A"][l], TR["A"][l] = D.ranks_of(mv, D.wi[D.VAL]).cpu(), D.ranks_of(mt, D.wi[D.TE]).cpu()
    SR["A"][l] = D.ranks_of(apply_M(m, xs), D.wi[D.TRS]).cpu()
    for jn, J in (("Jgen", JG), ("Jrel", JR)):
        VR[jn][l] = D.ranks_of(xv @ J[l].T, D.wi[D.VAL]).cpu()
        TR[jn][l] = D.ranks_of(xt @ J[l].T, D.wi[D.TE]).cpu()
    jv, jt = unit(xv @ JG[l].T), unit(xt @ JG[l].T)
    for lam in LAMS:
        VR[f"mix{lam}|A"][l] = D.ranks_of(jv + lam * unit(mv), D.wi[D.VAL]).cpu()
        TR[f"mix{lam}|A"][l] = D.ranks_of(jt + lam * unit(mt), D.wi[D.TE]).cpu()
    log(f"L{l} cfg {cfg} val10 {z[0]:.4f} | test top10 A {t10(TR['A'][l]):.4f} Jgen {t10(TR['Jgen'][l]):.4f} "
        f"Jrel {t10(TR['Jrel'][l]):.4f} mix0.3 {t10(TR['mix0.3|A'][l]):.4f} | |Wr| mean {m['Wr'].abs().mean():.2e} "
        f"max {m['Wr'].abs().max():.2e}; z std {((xv - m['mx']) @ m['P']).std(0).mean():.3f}")
    del X, xv, xt, xs, m
    torch.cuda.empty_cache()
torch.save(dict(VR=dict(VR), TR=dict(TR), SR=dict(SR), te_rows=D.TE.cpu(), val_rows=D.VAL.cpu(),
                trs_rows=D.TRS.cpu(), te_style=D.te_sty), f"{OUT}/base_ranks{'' if layers == READ else '_partial'}.pt")
res = dict(choice={str(k): v for k, v in choice.items()},
           layers={n: {str(l): [round(t10(TR[n][l]), 4), round(t1(TR[n][l]), 4)] for l in layers}
                   for n in ["A", "Jgen", "Jrel"] + [f"mix{lam}|A" for lam in LAMS]})
if layers == READ:
    vm = {lam: t10(any7(VR[f"mix{lam}|A"])) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    res.update(mix_val=vm, lam=lam, any7={n: [round(t10(any7(TR[n])), 4), round(t1(any7(TR[n])), 4)]
                                          for n in ["A", "Jgen", "Jrel", f"mix{lam}|A"]})
    ref = json.load(open(f"{R}/.hf27/mp1/fit_results.json"))
    res["mp1_ref_any7"] = {k: ref["T_any7"][k] for k in ["M_all", "Jgen|last", "Jrel|last", "mix|Jgen+M_all (last)"]}
    log(f"any7 {res['any7']} | mp1 {res['mp1_ref_any7']}")
json.dump(res, open(f"{RES}/sanity{'' if layers == READ else '_partial'}.json", "w"), indent=1)
log("SANITY_DONE")
