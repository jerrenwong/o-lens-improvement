"""xm1 copy of auto/make_split.py / make_split27.py for the new models (same rule, unchanged):
fit = the bench's own fit split (jpp_cli fit:0 over the model's scoreable items); held-out items split per task by
sha1(item name) order alternating dev, test, dev, ...  Writes auto/split_<tag>.json.
    PYTHONPATH=src python scripts/xm1/make_split_x.py <tag>"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from olens.readout import Bench

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHES = {t: f".hf27/xm1/bench_{t}.pt" for t in sys.argv[1:]}


def sha(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()


for tag, path in BENCHES.items():
    b = Bench.load(ROOT / path)
    keys = list(zip(b.eval_slugs, b.item_names, strict=True))
    out = {"dev": [], "test": [], "fit": [list(keys[i]) for i in b.index("fit")]}
    held = [keys[i] for i in b.index("held-out")]
    for task in sorted({e for e, _ in held}):
        items = sorted((k for k in held if k[0] == task), key=lambda k: (sha(k[1]), k[1]))
        for j, k in enumerate(items):
            out["dev" if j % 2 == 0 else "test"].append(list(k))
    for s in out:
        out[s].sort()
    (ROOT / f"auto/split_{tag}.json").write_text(json.dumps(out, indent=0))
    print(f"== {tag} ({b.model})")
    tasks = sorted(set(b.eval_slugs))
    print(f"  {'split':6s} " + " ".join(f"{t:>12s}" for t in tasks) + f" {'total':>7s}")
    for s in ("fit", "dev", "test"):
        c = Counter(e for e, _ in out[s])
        print(f"  {s:6s} " + " ".join(f"{c[t]:12d}" for t in tasks) + f" {len(out[s]):7d}")
