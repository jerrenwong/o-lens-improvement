"""print th1 report tables from analysis.json / causal.json (no torch)."""
import json, sys
B = "/workspace/o-lens-improvement/.hf27/th1"
A = json.load(open(f"{B}/analysis.json"))
print("asked (test):", A["asked_test"])
print("\nLayer | probe t1/t5 | classmean t1/t5 | SAY J++ t1/t5 | SAY J t1/t5 | logit t1/t5 | lam | offset/resid norm")
for L, r in A["layers"].items():
    f = lambda k: f"{r[k]['top1']:.3f}/{r[k]['top5']:.3f}"
    print(f"L{L} | {f('probe')} | {f('classmean')} | {f('say_jpp')} | {f('say_jlens')} | {f('logit_lens')} | {r['probe_lam']} | {r['offset_norm_median']:.1f}/{r['resid_norm_median']:.1f}")
print("\nGeometry (centred: matched / mismatched / ident ; raw: matched / mismatched / ident)")
for key in ["THINKa_vs_SAY", "THINKb_vs_SAY", "THINKa_vs_SAYJ", "THINKa_vs_EMB", "THINKb_vs_EMB", "THINKa_vs_UNEMB", "THINKb_vs_UNEMB", "THINKa_vs_THINKb", "SAY_vs_UNEMB", "SAY_vs_SAYJ"]:
    print(key)
    for L, r in A["layers"].items():
        g = r["geometry"][key]; c, w = g["centred"], g["raw"]
        print(f"  L{L}: c {c['matched']:+.3f} {c['mismatched']:+.3f} {c['ident']:.3f} | raw {w['matched']:+.3f} {w['mismatched']:+.3f} {w['ident']:.3f}")
print("\nOverlap top-5 (share probe hits also SAY / share SAY hits also probe), by kind probe/say/logit top5 at some layers")
for L, r in A["layers"].items():
    o = r["overlap_top5"]
    print(f"L{L}: {o['share_probe_hits_also_say']:.3f} / {o['share_say_hits_also_probe']:.3f}  (probe {o['probe_hits']}, say {o['say_hits']}, both {o['both']})")
for L in ("8", "16", "24", "28"):
    print(f"L{L} by kind:", A["layers"][L]["by_kind"])
try:
    C = json.load(open(f"{B}/causal.json"))
    print("\nCausal: cond | says X | sum p(X) | ask X top1/top5 | median rank X | says Y | sum p(Y) | ask Y top1/top5")
    for k, v in C["agg"].items():
        print(f"{k:16s} | {v['says_x']:.2f} | {v['px']:.3f} | {v['ask_x_top1']:.2f}/{v['ask_x_top5']:.2f} | {v['ask_x_mrank']} | {v['says_y']:.2f} | {v['py']:.3f} | {v['ask_y_top1']:.2f}/{v['ask_y_top5']:.2f}")
except FileNotFoundError:
    pass
