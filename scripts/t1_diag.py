"""t1 diagnostics per lens (Qwen3.5-2B), extending scripts/r1_diag.py:
 (1) share of filtered top-10 slots starting with '@' on association prompts per readout layer (FIT+DEV items, as r1_diag);
 (2) decoded top-3 singular output directions (both signs, filtered top-6) at layers 15/18/21;
 (3) on DEV association and DEV poetry prompts: the 8 most common UNFILTERED top-1 tokens per layer (+ the share of
     top-1 tokens that are punctuation-excluded ids / in the fit pool's top-200 list).
    cd /workspace/o-lens-improvement && python scripts/t1_diag.py name=path.pt ... > log   (writes runs/t1_2b/diag.json)"""
import collections, json, sys
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
sys.path.insert(0, "/workspace/o-lens-improvement/src")
from olens.readout import Bench  # noqa: E402

R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-2B"; LAYERS = [3, 6, 9, 12, 15, 18, 21]
t.set_grad_enabled(False)
lenses = dict(a.split("=", 1) for a in sys.argv[1:])
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-2b.pt")
sp = json.load(open(f"{R}/auto/split_2b.json"))
test = {tuple(x) for x in sp["test"]}
fitdev = {tuple(x) for k in ("fit", "dev") for x in sp[k]}
dev = {tuple(x) for x in sp["dev"]}
key = lambda i: (b.eval_slugs[i], b.item_names[i])
idx_at = [i for i in range(len(b.item_names)) if key(i) in fitdev and b.eval_slugs[i] == "association"]
idx_dev = {e: [i for i in range(len(b.item_names)) if key(i) in dev and b.eval_slugs[i] == e] for e in ("association", "poetry")}
assert all(key(i) not in test for i in idx_at + sum(idx_dev.values(), []))
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
top200 = t.zeros(Vn, dtype=t.bool).cuda(); top200[json.load(open(f"{R}/runs/t1_2b/content_rule.json"))["top200"]] = True
at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()
NEG = float("-inf")
print(f"association fit+dev prompts: {len(idx_at)}; dev association {len(idx_dev['association'])}, dev poetry {len(idx_dev['poetry'])}")
out = {}
for name, path in lenses.items():
    J = load_lens_file(path, hf_model_name=M).jacobians_L_dict_FN
    row = {"at_share": {}, "top1": {}}
    for L in LAYERS:
        v = b.resid[L][idx_at].cuda().float() @ J[L].cuda().float().T
        tp = model.unembed(v).float().masked_fill(punct, NEG).topk(10, -1).indices
        row["at_share"][L] = float(at[tp].float().mean())
    print(f"@share {name:14s} " + " ".join(f"L{L} {row['at_share'][L]:.2f}" for L in LAYERS), flush=True)
    for L in (15, 18, 21):
        U, S, _ = t.linalg.svd(J[L].cuda().float(), full_matrices=False)
        def dec(u):
            return " ".join(repr(strs[j].strip()) for j in model.unembed(u[None]).float().masked_fill(punct, NEG).topk(6, -1).indices[0].tolist())
        print(f"  {name} L{L} sigma1..3/median {[round(float(x / S.median()), 1) for x in S[:3]]}")
        for k in range(3):
            print(f"     u{k+1}+: {dec(U[:, k])}   |   u{k+1}-: {dec(-U[:, k])}")
    for e, ix in idx_dev.items():
        row["top1"][e] = {}
        print(f"  {name} dev {e}: 8 most common unfiltered top-1 tokens (count/{len(ix)}), share punct-excluded, share top200")
        for L in LAYERS:
            v = b.resid[L][ix].cuda().float() @ J[L].cuda().float().T   # all positions of the item, [n, ...] as bench stores
            t1 = model.unembed(v).float().argmax(-1).flatten()
            c = collections.Counter(strs[j] for j in t1.tolist())
            row["top1"][e][L] = {"common": c.most_common(8), "punct": float(punct[t1].float().mean()), "top200": float(top200[t1].float().mean())}
            print(f"    L{L:2d} punct {row['top1'][e][L]['punct']:.2f} top200 {row['top1'][e][L]['top200']:.2f} | " + "  ".join(f"{s!r}:{n}" for s, n in c.most_common(8)))
    out[name] = row
    print(flush=True)
json.dump(out, open(f"{R}/runs/t1_2b/diag.json", "w"), indent=1)
