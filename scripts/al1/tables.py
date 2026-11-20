"""al1: print report tables from fit_results.json (no torch)."""
import json, sys
r = json.load(open(sys.argv[1]))
N = ["M1", "M2", "M3", "M4", "mix", "J++", "J", "logit", "G_g1", "E"]
print("counts", r["counts"], "best_M", r["best_M"], "best layers", r["best_layer"], "mix lam", r["mix_lam"], r["mix_val"])
print("val any-layer top10", r["val_anylayer_top10"])
print("\n== validation choices per layer (cfg, val top10)")
for l, ch in r["choice"].items():
    print(f"L{l} | " + " | ".join(f"{n} {ch[n]['cfg']} {ch[n]['val_top10']}" for n in ("M1", "M2", "M3", "M4")))
print("\n== test words, per layer top10/top1")
print("layer | " + " | ".join(N))
for l, row in r["layers"].items():
    print(f"L{l} | " + " | ".join(f"{row[n]['top10']:.3f}/{row[n]['top1']:.3f}" if n in row else "-" for n in N))
print("any7 | " + " | ".join(f"{r['any_layer'][n]['top10']:.3f}/{r['any_layer'][n]['top1']:.3f}" for n in N))
print("shuf any7", r["any_layer"].get("M_shuf"))
print("best single layer (test)", {n: r["best_single_layer_test"][n] for n in N})
print("\n== k-sentence means (any7 top10/top1 ; best layer top10)")
print("k (n) | " + " | ".join(N))
for k, row in r["kmean"].items():
    n0 = row[N[0]]["n"]
    print(f"k={k} ({n0}) | " + " | ".join(
        f"{row[n]['any']['top10']:.3f}/{row[n]['any']['top1']:.3f} ; L{row[n]['best'][0]} {row[n]['best'][1]['top10']:.3f}"
        for n in N))
print("\n== learning curve", r["best_M"])
for n, v in r["learning_curve"].items():
    print(n, v)
print("\n== complementarity", r["complement"], r["complement_bestlayer"])
print("\n== examples")
for e in r["examples"]:
    print(f"[{e['word']}] {e['sent']}\n   M rank {e['M_rank']}: {', '.join(e['M'])}\n   J++ rank {e['J_rank']}: {', '.join(e['J'])}")
