"""ex2: write the DEV lens specs (runs/ex2/specs_<model>.json) for S3, S4, S6, S7 from scales.json, gram.json, s4_*.json.
Every coefficient here was computed from FIT activations / Frobenius algebra / FIT scores only.

S3  <fit>_b<beta>   : L + beta * s_L * Jbar            (s_L = RMS ||L h|| / RMS ||Jbar h|| on FIT activations, per layer)
    <fit>_perp_g<g> : (L - a Jbar) + g * s_perp * Jbar  (a = Frobenius projection coefficient of L on Jbar;
                       s_perp = RMS ||(L - aJbar) h|| / RMS ||Jbar h||; L itself sits at g = gamma_L = a / s_perp)
    <fit>_L, <fit>_pool
S4  <fit>_s4single, _s4content, _s4P (alpha chosen on FIT), _s4P0 (P alone, reference), _s4sum0
S6  <m>rel_jpp, <m>rel_jl, <m>rel_jl_b<beta> = JPP + beta s_jl J-lens, <m>rel_pool_b<beta> (pooled LRP lens of the
    main fit), <m>rel_{jl,pool}_perp0 = JPP minus its Frobenius projection on J-lens / pooled
S7  9b_64a_drop_L<l>_e<e> : L with w_e set to 0 at layer l only (clusters differ per layer)
"""
import json
import sys

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
from ex2_common import FITS, LAYERS, MAIN, OUT  # noqa: E402

BETAS = [-1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8]
GAMMAS = [0, 0.25, 0.5, 1, 2, 4, 8]
REL_BETAS = [-1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8]
scales = json.load(open(f"{OUT}/scales.json"))
gram = json.load(open(f"{OUT}/gram.json"))["fits"]


def spec(m, fn):
    return {"layers": {str(L): fn(L) for L in LAYERS[m]}}


for m in ("9b", "2b"):
    S = {}
    fits = [f for f, c in FITS.items() if c["m"] == m and f != "9b_w2"]
    for f in fits:
        sc = scales[f]
        S[f"{f}_L"] = spec(m, lambda L: [[f"L:{f}", 1.0]])
        if f == MAIN[m]:  # repeat of L in a separate standalone `python auto/evaluate.py` process (noise reference)
            S[f"{f}_Lrep"] = spec(m, lambda L: [[f"L:{f}", 1.0]])
        S[f"{f}_pool"] = spec(m, lambda L: [[f"J:{f}", 1.0]])
        for b in BETAS:
            S[f"{f}_b{b}"] = spec(m, lambda L, b=b: [[f"L:{f}", 1.0], [f"J:{f}", b * sc[str(L)]["s_L"]]])
        for g in GAMMAS:
            S[f"{f}_perp_g{g}"] = spec(m, lambda L, g=g: [[f"L:{f}", 1.0],
                                                          [f"J:{f}", -sc[str(L)]["a_fro"] + g * sc[str(L)]["s_perp"]]])
        try:
            s4 = json.load(open(f"{OUT}/s4_{f}.json"))
        except FileNotFoundError:
            print("no s4 for", f)
            continue
        lay = s4["layers"]

        def ecoefs(c8, jcoef):
            return [[f"E:{f}:{e}", float(c8[e])] for e in range(8)] + [[f"J:{f}", float(jcoef)]]

        for fam in ("single", "content", "P"):
            if f"alpha_{fam}" not in s4:
                continue
            a = s4[f"alpha_{fam}"]
            S[f"{f}_s4{fam}"] = spec(m, lambda L, fam=fam, a=a: ecoefs(lay[str(L)][f"c_{fam}"], -a * lay[str(L)][f"a_{fam}"]))
        S[f"{f}_s4P0"] = spec(m, lambda L: ecoefs(lay[str(L)]["c_P"], 0.0))
        S[f"{f}_s4sum0"] = spec(m, lambda L: ecoefs(lay[str(L)]["sum0"]["w"], 0.0))
        if f == "9b_64a":  # S7: clusters are per layer, so drop one (layer, expert) at a time
            for L0 in LAYERS[m]:
                for e in range(8):
                    S[f"{f}_drop_L{L0}_e{e}"] = spec(m, lambda L, e=e, L0=L0: (
                        ecoefs([0.0 if k == e else gram[f][str(L)]["w"][k] for k in range(8)], 0.0) if L == L0
                        else [[f"L:{f}", 1.0]]))
    # released
    rs = scales[f"released_{m}"]
    pf = MAIN[m]
    S[f"{m}rel_jpp"] = spec(m, lambda L: [["RJPP", 1.0]])
    S[f"{m}rel_jl"] = spec(m, lambda L: [["RJ", 1.0]])
    for b in REL_BETAS:
        S[f"{m}rel_jl_b{b}"] = spec(m, lambda L, b=b: [["RJPP", 1.0], ["RJ", b * rs[str(L)]["s_jl"]]])
        S[f"{m}rel_pool_b{b}"] = spec(m, lambda L, b=b: [["RJPP", 1.0], [f"J:{pf}", b * rs[str(L)]["s_pool"]]])
    S[f"{m}rel_jl_perp0"] = spec(m, lambda L: [["RJPP", 1.0], ["RJ", -rs[str(L)]["a_jl"]]])
    S[f"{m}rel_pool_perp0"] = spec(m, lambda L: [["RJPP", 1.0], [f"J:{pf}", -rs[str(L)]["a_pool"]]])
    json.dump(S, open(f"{OUT}/specs_{m}.json", "w"), indent=0)
    print(m, len(S), "specs")
