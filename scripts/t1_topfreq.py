"""t1: the label-free 'content next token' rule. Counts token ids over the 1,000-prompt raw WikiText fit pool
(runs/r1_2b/prompts_raw.json), tokenized exactly as the fit sees them (model.encode: tokenizer, truncation,
max_length 128), and writes runs/t1_2b/content_rule.json:
  top200      the 200 most frequent ids
  content     bool per vocab id: decoded token contains a letter (str.isalpha on any char) AND id not in top200
Usage (pod): /root/jpp_lens/.venv/bin/python scripts/t1_topfreq.py"""
import collections, json
from transformers import AutoTokenizer
R = "/workspace/o-lens-improvement"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
prompts = json.load(open(f"{R}/runs/r1_2b/prompts_raw.json"))
assert len(prompts) >= 1000
prompts = prompts[:1000]
cnt = collections.Counter()
ntok = 0
for p in prompts:
    ids = tok(p, truncation=True, max_length=128).input_ids
    cnt.update(ids); ntok += len(ids)
top = [i for i, _ in cnt.most_common(200)]
V = max(len(tok), max(cnt) + 1)
strs = [tok.decode([i]) for i in range(V)]
content = [any(c.isalpha() for c in s) and (i not in set(top)) for i, s in enumerate(strs)]
share_top = sum(cnt[i] for i in top) / ntok
share_content = sum(c for i, c in cnt.items() if content[i]) / ntok
json.dump({"top200": top, "top200_str": [strs[i] for i in top], "vocab": V, "content": content,
           "pool_tokens": ntok, "pool_share_top200": share_top, "pool_share_content": share_content},
          open(f"{R}/runs/t1_2b/content_rule.json", "w"))
print(f"tokens {ntok}  vocab {V}  share in top200 {share_top:.3f}  share content {share_content:.3f}")
print("top200:", " | ".join(repr(strs[i]) for i in top))
