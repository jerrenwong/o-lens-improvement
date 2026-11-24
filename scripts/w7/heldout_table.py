import json
src = open("/workspace/o-lens-improvement/scripts/f4_page_numbers.py").read().split('P = {"w1"')[0]
g = {}; exec(src, g)
out = {}
for t in ("9b", "2b"):
    for kind in ("test", "v4"):
        N = {"JPP0": f"w1_{t}_JPP0_f4_{kind}"}
        for c, a, j in [("C1", f"w1_{t}_none_f4_{kind}", f"w1_{t}_JPP_f4_{kind}"), ("C3", f"w2ho_{t}_ce_none_f4_{kind}", f"w2ho_{t}_ce_JPP_f4_{kind}")] + \
                       [(c, f"w7ho_{t}_{m}_none_f4_{kind}", f"w7ho_{t}_{m}_JPP_f4_{kind}") for c, m in (("C2", "all"), ("C4", "mergeall"), ("C5", "merge"))]:
            N[c] = a; N["J+" + c] = j
        s = {r: g["S"](n) for r, n in N.items()}
        pairs = [(x + a, x + b) for x in ("", "J+") for a, b in (("C2", "C1"), ("C3", "C2"), ("C4", "C2"), ("C5", "C2"), ("C5", "C3"))]
        out[f"{t}_{kind}"] = dict(rows={r: [round(100*s[r]["pass10"], 1), round(100*s[r]["pass1"], 1)] for r in sorted(N)}, d=g["diffs"](N, pairs))
        print(t, kind, out[f"{t}_{kind}"]["rows"]); print("   ", {k: v for k, v in out[f"{t}_{kind}"]["d"].items() if k.startswith("J+")})
json.dump(out, open("/workspace/o-lens-improvement/runs/w7/heldout.json", "w"), indent=0)
