"""Qualitative + quantitative check: what is underneath the symbol/markup junk?
Ranks concepts with (i) the benchmark's punctuation filter and (ii) a "plain word" filter that also removes tokens
containing any non-alphanumeric character (’d, —he, @g, \\n, <br, <|im_end|> ...). FIT + DEV items only."""
import json, collections, sys
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]   # test never enters
J = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
def plain(s):
    u = s[1:] if s.startswith(" ") else s
    return len(u) > 0 and u.isalnum()
word = t.tensor([plain(s) for s in strs]).cuda()
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
print(f"vocab {Vn}: benchmark filter removes {int(punct.sum())}; plain-word filter removes {int((~word).sum())}")
cand_all = sorted({c for i in idx for ids in b.candidates[i].values() for c in ids})
print("concept candidate tokens removed by the plain-word rule:", sum(not plain(strs[c]) for c in cand_all), "of", len(cand_all),
      [strs[c] for c in cand_all if not plain(strs[c])][:12])
res = collections.defaultdict(dict)   # (lens, mask) -> (item, inter) -> best rank ; also per layer
perlayer = collections.defaultdict(lambda: collections.Counter())
tops = {}
for L in LAYERS:
    h = b.resid[L][idx].cuda().float()
    for name, Mx in (("J", J), ("JPP", P)):
        v = h @ Mx[L].cuda().float().T
        for s in range(0, len(idx), 32):
            lg = model.unembed(v[s:s+32]).float()
            for mname, mask in (("std", punct), ("word", ~word)):
                l2 = lg.masked_fill(mask, float("-inf"))
                top = l2.topk(10, -1).indices
                for r in range(l2.shape[0]):
                    i = idx[s + r]
                    tops[(name, mname, L, i)] = [strs[x] for x in top[r].tolist()]
                    for inter, ids in b.candidates[i].items():
                        ids2 = [c for c in ids if not mask[c]]
                        rk = int((l2[r] > l2[r, ids2].max()).sum()) + 1 if ids2 else 10**9
                        key = (i, inter); d = res[(name, mname)]
                        d[key] = min(d.get(key, 10**9), rk)
                        perlayer[(name, mname, L)][b.eval_slugs[i]] += rk <= 10
def macro(d, k, split):
    item = collections.defaultdict(list)
    for (i, inter), r in d.items():
        if part[(b.eval_slugs[i], b.item_names[i])] == split: item[i].append(r <= k)
    ev = collections.defaultdict(list)
    for i, hs in item.items(): ev[b.eval_slugs[i]].append(sum(hs) / len(hs))
    return sum(sum(v) / len(v) for v in ev.values()) / len(ev), {e: round(sum(v) / len(v), 3) for e, v in sorted(ev.items())}
for split in ("fit", "dev"):
    print(f"\n== {split}: macro pass@10 / pass@1 and per-task pass@10")
    for name in ("J", "JPP"):
        for mname in ("std", "word"):
            p10, per = macro(res[(name, mname)], 10, split); p1, _ = macro(res[(name, mname)], 1, split)
            print(f"  {name:3s} {mname:4s}  {p10:.4f} / {p1:.4f}   {per}")
print("\n== hits (rank<=10) per layer, fit+dev pairs, by task: std -> word")
for name in ("J", "JPP"):
    for task in ("association", "multilingual", "poetry", "multihop", "typo"):
        print(f"  {name:3s} {task:12s} " + "  ".join(f"L{L}:{perlayer[(name,'std',L)][task]}->{perlayer[(name,'word',L)][task]}" for L in LAYERS))
want = ["einstein", "city-h", "china-h", "space-h", "couplet-jar-car", "couplet-sea-free"]
print("\n== examples under the plain-word filter (top 8)")
for i in idx:
    if b.item_names[i] in want:
        print(f"[{b.eval_slugs[i]}/{b.item_names[i]}] concept {list(b.candidates[i])}")
        for L in (12, 20, 24, 28):
            for name in ("J", "JPP"):
                print(f"   L{L} {name:3s}: " + " | ".join(s.strip() for s in tops[(name, 'word', L, i)][:8]))
