"""sub2 helper: meta sizes and overlap of pilot al2 rows with meta rows (by word, style, sentence)."""
import json, collections, torch
R = "/workspace/o-lens-improvement"
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
print(meta.keys())
n = len(meta["words"]); print("n", n, "words", len(meta["role"]), collections.Counter(meta["split"]), collections.Counter(meta["style"]))
key = collections.defaultdict(list)
for i, (w, k, s) in enumerate(zip(meta["words"], meta["style"], meta["sents"])):
    key[(w, k, s)].append(i)
print("dup keys", sum(len(v) > 1 for v in key.values()))
a = [json.loads(l) for l in open(f"{R}/.hf27/sub1/pilot.jsonl")]
a = [r for r in a if r["src"] != "new"]
hit = [r for r in a if (r["word"], int(r["style"][1:]) - 1, r["sent"]) in key]
hit2 = [r for r in a if r["word"] in meta["role"]]
print("pilot al2 rows", len(a), "in meta exact", len(hit), "word in meta", len(hit2))
