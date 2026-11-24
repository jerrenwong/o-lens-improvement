"""w1: per-model breakdown of the f2 mask (no weights; tokenizer + bench only). PYTHONPATH=src, HF_HOME as needed."""
import importlib.util, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
ROOT = Path("/workspace/o-lens-improvement"); sys.path.insert(0, str(ROOT / "src"))
from olens.readout import Bench
s = importlib.util.spec_from_file_location("w1_filters", ROOT / "auto/filters.py"); F = importlib.util.module_from_spec(s); s.loader.exec_module(F)
model, bench = sys.argv[1], sys.argv[2]; V = AutoConfig.from_pretrained(model).get_text_config().vocab_size
tok = AutoTokenizer.from_pretrained(model)
FL = F.Filters(tok, V, [Bench.load(ROOT / bench)])
f1, f2, f3 = FL.get("f1"), FL.get("f2"), FL.get("f3")
new = sorted(set(f2.excluded_ids.tolist()) - set(f1.excluded_ids.tolist()))
print(model, "V", V, "len(tok)", len(tok), "f1", len(f1.excluded_ids), "f2", len(f2.excluded_ids), "f3 groups", f3.n_groups, f2.parts)
print("  newly masked at f2:", len(new), "e.g.", repr([FL.texts[i] for i in new[:50]]))
sp = F.spared_ids(FL.benches)
print("  spared that are special/@/no-alnum:", [FL.texts[i] for i in sorted(sp) if (i in F.special_ids(tok)) or "@" in FL.texts[i] or not any(c.isalnum() for c in FL.texts[i])][:40])
