"""w11 junk check (pod): share of DEV items whose UNFILTERED top-1 token is junk (f4 token rule: no letter/digit,
special/added, contains @ & < >), per readout layer and mean over layers, from W11_DUMP files of evaluator runs.
    python scripts/w11/w11_junk.py <model> <dump.pt>...  -> prints a table, appends runs/w11/junk.json"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(R / "scripts/w11"))
from transformers import AutoTokenizer  # noqa: E402
from w11_accum import junk_mask  # noqa: E402

model = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model)
res = {}
junk = None
for p in sys.argv[2:]:
    d = torch.load(p)
    if junk is None:
        V = int(max(int(v.max()) for v in d.values())) + 1
        junk = junk_mask(tok, max(V, 248320))
    per = {}
    for l, v in sorted(d.items()):
        n = v.shape[0] // 2  # the evaluator unembeds each row twice (filtered, unfiltered)
        per[l] = float(junk[v[:n, 0]].float().mean())
    res[Path(p).stem] = dict(per_layer=per, mean=sum(per.values()) / len(per))
    print(Path(p).stem, f"mean {res[Path(p).stem]['mean']:.3f}", {l: round(x, 3) for l, x in per.items()})
f = R / "runs/w11/junk.json"
old = json.loads(f.read_text()) if f.exists() else {}
old.update(res)
f.write_text(json.dumps(old, indent=1))
