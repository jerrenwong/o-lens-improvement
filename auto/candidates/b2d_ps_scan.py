"""b2d patchscope (idea 6), S alone with the 'repeat' prompt (h written at '?', replace).
build() also prints a FIT-split scan over prompts / modes / J++ mixing weights (diagnostic only)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2d_common", Path(__file__).with_name("b2d_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

CONFIGS = [("the", "replace"), ("repeat", "replace"), ("repeat", "add"), ("concept", "replace"),
           ("concept_ph", "replace")]
LAMS = [0.5, 1.0, 2.0]


def build(ctx):
    L, R = ctx.readout_layers, ctx.fit.resid
    jpp = {l: R[l] @ ctx.JPP[l].T for l in L}
    C.report(ctx, "J++", jpp)
    print("[fit] per-layer hits J++", C.per_layer_hits(ctx, jpp), flush=True)
    for pname, mode in CONFIGS:
        P = C.Patcher(ctx.model, C.PROMPTS[pname], mode)
        S = {l: P(R[l], l) for l in L}
        C.report(ctx, f"S {pname}/{mode}", S)
        print(f"[fit] per-layer hits S {pname}/{mode}", C.per_layer_hits(ctx, S), flush=True)
        for lam in LAMS:
            C.report(ctx, f"J++ + {lam} S {pname}/{mode}",
                     {l: C.unit(jpp[l]) + lam * C.unit(S[l]) for l in L})
    return C.PatchLens(ctx.JPP, C.Patcher(ctx.model, C.PROMPTS["repeat"], "replace"), None)
