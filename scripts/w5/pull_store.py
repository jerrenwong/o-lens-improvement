"""W5: fill the residual store from the published dataset only (no model, no forward pass).
For each family, the plan items whose cells the dataset holds under today's key are pulled into
the store (wsbench.produce.store.cells_for with backend=None); items in neither place are listed.
Run with the wsbench venv: python pull_store.py <store> fast|full fam1,fam2,...
Writes <store>/_coverage_<tier>.json."""
import json
import sys
from pathlib import Path

from wsbench import readplan
from wsbench.produce import dataset, store
from wsbench.produce.backend import DEFAULT_REVISION

root, tier, fams = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
fast = tier == "fast"
where = store.Store(root, DEFAULT_REVISION, "fp32")
subject = store.Subject("Qwen/Qwen3.6-27B", DEFAULT_REVISION, "fp32", "cpu")
pull = dataset.source("agu18dec/workspacebench_residuals", "", "fp32", DEFAULT_REVISION)
cov = {}
for fam in fams:
    specs = readplan.plan(fam, reads="full", fast=fast)
    located = [store.locate(s, subject.tokenizer) for s in specs]
    stored = where.has(fam, {at.spec.id: at.key for at in located})
    ok = [s for s, at in zip(specs, located) if at.spec.id in stored or pull.folder_of(at) is not None]
    miss = [s.id for s in specs if s not in ok]
    cells, counts = store.cells_for(where, fam, ok, subject.tokenizer, pull=pull, backend=None)
    n_plan = sum(len(at.rows) for at in located)
    cov[fam] = dict(items=len(specs), items_stored=len(ok), items_missing=miss, cells_plan=n_plan,
                    cells_stored=len(cells))
    print(counts.line(fam), f"| items {len(ok)}/{len(specs)}, cells {len(cells)}/{n_plan}", flush=True)
p = Path(root) / f"_coverage_{tier}.json"
old = json.loads(p.read_text()) if p.exists() else {}
old.update(cov)
p.write_text(json.dumps(old, indent=1))
print("files downloaded:", len(pull.downloaded))
