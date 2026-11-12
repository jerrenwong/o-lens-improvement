"""Split pass@10 by readout position (< 16 = never seen in training vs >= 16). Tokenizer only."""
import json, sys
import pandas as pd
from transformers import AutoTokenizer
sys.path.insert(0, "/root/jpp_lens/src")
from lens_evals.readout_evals.readout_eval_items import READOUT_EVAL_SPECS, load_readout_eval_items, MACRO_EVALS
from lens_evals.readout_evals.readout_evals import readout_position
import os
tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "Qwen/Qwen3.5-2B"))
items = load_readout_eval_items("/root/jpp_lens/data/jlens/evaluations", slugs=MACRO_EVALS)
pos = {}
for it in items:
    ids = tok(it.prompt if isinstance(it.prompt, str) else tok.apply_chat_template(it.prompt, tokenize=False), return_tensors="pt").input_ids
    pos[(it.eval_slug, it.name)] = readout_position(ids, tok, READOUT_EVAL_SPECS[it.eval_slug].readout_rule)
df = pd.read_csv(sys.argv[1])
best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
best["hit"] = best["rank"] <= 10
best["pos"] = [pos.get((e, i), -1) for e, i in zip(best["eval"], best["item"])]
best["late"] = best["pos"] >= 16
print("readout positions per eval:", best.drop_duplicates(["eval","item"]).groupby("eval")["pos"].describe()[["min","50%","max"]].to_string())
t = best.groupby(["late", "lens"])["hit"].mean().unstack("late")
t.columns = ["pos<16", "pos>=16"]
t["n_items<16"] = best[~best.late].drop_duplicates(["eval","item"]).shape[0]
print(t.round(3).to_string())
