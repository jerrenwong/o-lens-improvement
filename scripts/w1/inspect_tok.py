"""w1: inspect tokenizer special/added tokens and bench task slugs (no model weights)."""
import sys, collections, torch
from transformers import AutoTokenizer
name, bench = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(name)
print("len", len(tok), "special", len(tok.all_special_ids))
ad = tok.added_tokens_decoder
print("added", len(ad), "special-flag", sum(a.special for a in ad.values()))
ns = [(i, a.content) for i, a in ad.items() if not a.special]
print("non-special added (first 40):", ns[:40])
d = torch.load(bench, map_location="cpu", weights_only=True)
print("tasks", collections.Counter(d["eval_slugs"]), "excluded", len(d["excluded_ids"]))
