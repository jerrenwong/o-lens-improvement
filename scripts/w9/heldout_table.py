import json, sys, importlib.util
sp = importlib.util.spec_from_file_location("f4", "/workspace/o-lens-improvement/scripts/f4_page_numbers.py")
src = open(sp.origin).read().split('P = {"w1"')[0]
g = {}; exec(src, g)
out = {}
for t in ("9b", "2b"):
    for kind in ("test", "v4"):
        N = {"JPP0": f"w1_{t}_JPP0_f4_{kind}", "ridge": f"w1_{t}_none_f4_{kind}", "JPP+ridge": f"w1_{t}_JPP_f4_{kind}",
             "ce28": f"w2ho_{t}_ce_none_f4_{kind}", "JPP+ce28": f"w2ho_{t}_ce_JPP_f4_{kind}",
             "ce98": f"w9ho_{t}_lin_ce_98k_none_f4_{kind}", "JPP+ce98": f"w9ho_{t}_lin_ce_98k_JPP_f4_{kind}"}
        s = {r: g["S"](n) for r, n in N.items()}
        out[f"{t}_{kind}"] = dict(rows={r: [round(100*s[r]["pass10"], 1), round(100*s[r]["pass1"], 1)] for r in N},
            d=g["diffs"](N, [("JPP+ce98", "JPP+ce28"), ("ce98", "ce28"), ("JPP+ce98", "JPP+ridge"), ("JPP+ce98", "JPP0"), ("ce98", "JPP0")]))
        print(t, kind, out[f"{t}_{kind}"])
json.dump(out, open("/workspace/o-lens-improvement/runs/w9/heldout.json", "w"), indent=0)
