"""w1: compare expert-weight JSONs: per layer signed shares, init/step, train pass, cosine to a reference."""
import json, math, sys
ref = json.load(open(sys.argv[1]))["by_layer"]
for p in sys.argv[2:]:
    w = json.load(open(p))["by_layer"]
    print(f"== {p.split('/')[-1]} vs {sys.argv[1].split('/')[-1]}")
    for l in ref:
        a, b = ref[l]["weights_unit_l2"], w[l]["weights_unit_l2"]
        cos = sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        md = max(abs(x - y) for x, y in zip(a, b))
        print(f"L{l:>2} cos {cos:+.4f} maxdiff {md:.1e} | {w[l]['init']:>11s} s{w[l]['step']:<3d} tr {w[l]['train_macro_pass']:.3f}"
              f" (ref {ref[l]['init']} s{ref[l]['step']} {ref[l]['train_macro_pass']:.3f}) | "
              + " ".join(f"{x:+.2f}" for x in w[l]["signed_shares"]))
