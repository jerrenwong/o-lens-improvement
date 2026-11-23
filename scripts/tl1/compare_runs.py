"""tl1 reproduction check: run A vs reference run B (auto/runs/<name>): pass@10/1 (filtered, unfiltered) and every
per-row rank in readout_{filtered,unfiltered}.csv must be identical.   python compare_runs.py <new> <ref> [...pairs]"""
import csv
import json
import sys
from pathlib import Path

R = Path("/workspace/o-lens-improvement/auto/runs")
ok_all = True
for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    sa, sb = (json.loads((R / n / "summary.json").read_text()) for n in (a, b))
    ok = all(sa[k] == sb[k] for k in ("pass10", "pass1", "pass10_unf", "pass1_unf", "n_items"))
    msg = []
    for kind in ("filtered", "unfiltered"):
        rows = []
        for n in (a, b):
            with open(R / n / f"readout_{kind}.csv") as fh:
                rows.append({(r["eval"], r["item"], r["intermediate"], r["layer"]): r["rank"] for r in csv.DictReader(fh)})
        diff = sum(rows[0].get(k) != v for k, v in rows[1].items()) + abs(len(rows[0]) - len(rows[1]))
        msg.append(f"{kind} rows {len(rows[0])}/{len(rows[1])} differ {diff}")
        ok &= diff == 0
    ok_all &= ok
    print(f"REPRO {a} vs {b}: {'EXACT' if ok else 'DIFFERENT'}; pass10 {sa['pass10']:.4f}/{sb['pass10']:.4f} "
          f"pass1 {sa['pass1']:.4f}/{sb['pass1']:.4f}; " + "; ".join(msg), flush=True)
print("ALL_EXACT" if ok_all else "NOT_ALL_EXACT")
