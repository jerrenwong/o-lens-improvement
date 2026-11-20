"""Split each model's held-out bench items into dev / test (about half each), stratified by task.

Deterministic: within a task, items are sorted by sha1 of the item name and alternate dev, test, dev, ...
Writes auto/split_9b.json and auto/split_2b.json: {"dev": [[eval, name], ...], "test": [...], "fit": [...]}.

    python auto/make_split.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from olens.readout import Bench

ROOT = Path(__file__).resolve().parent.parent
BENCHES = {"9b": "artifacts/bench/qwen3.5-9b.pt", "2b": "artifacts/bench/qwen3.5-2b.pt"}


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
