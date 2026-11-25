"""w11 coverage (pod): share of FIT / DEV concepts (item, intermediate) with at least one candidate id that received a
direction with N_X >= nmin; per-concept flags for DEV (used by w11_table.py for covered / uncovered scores).
    python scripts/w11/w11_cover.py 2b      -> runs/w11/cover_2b.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "src"))
from olens.readout import Bench  # noqa: E402

tag = sys.argv[1]
bench = Bench.load(R / f"artifacts/bench/qwen3.5-{tag}.pt")
split = json.loads((R / f"auto/split_{tag}.json").read_text())
meta = torch.load(R / f".hf27/w11/acc_{tag}_meta.pt", map_location="cpu")
N = dict(zip(meta["tok_ids"].tolist(), meta["N"].tolist()))
pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
out = {"n_tokens": len(N)}
for nmin in (1, 3, 10, 30, 100):
    o = {}
    for sp in ("fit", "dev"):
        flags = {}
        for k in split[sp]:
            i = pos[tuple(k)]
            for inter, ids in bench.candidates[i].items():
                flags[f"{bench.eval_slugs[i]}|{bench.item_names[i]}|{inter}"] = any(N.get(t, 0) >= nmin for t in ids)
        o[sp] = sum(flags.values()) / len(flags)
        per = {}
        for key, v in flags.items():
            per.setdefault(key.split("|")[0], []).append(v)
        o[sp + "_task"] = {t: sum(v) / len(v) for t, v in per.items()}
        if sp == "dev":
            o["dev_flags"] = flags
    out[nmin] = o
    print(nmin, {k: v for k, v in o.items() if k != "dev_flags"})
(R / f"runs/w11/cover_{tag}.json").write_text(json.dumps(out))
