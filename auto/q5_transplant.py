"""q5: all-layer column transplant for the 61 association items (FIT + DEV only; test never loaded into any statistic).
Is the concept present AT the passage's final period token?  Record block outputs at that token (source run) and
write them into the final period of a neutral filler passage inside the few-shot question prompt FS (target run);
read the next-token logits after 'One-word topic:'.  Hooks patch BLOCK OUTPUTS only (residual stream)."""
import json, collections, statistics
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
blocks = model.layers; emb = model._embed_tokens; NL = len(blocks)
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json")); RV = rd["vocab"]
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in part}
text = {k: "".join(RV[x] for x in it["toks"]) for k, it in meta.items()}
FS = ("Passage: The whistle blew and twenty-two players ran onto the wet grass while the crowd roared.\nOne-word topic: football\n\n"
      "Passage: She blew out the candles while everyone sang, then cut the first slice.\nOne-word topic: birthday\n\n"
      "Passage: He adjusted the telescope and wrote down the position of the red planet again.\nOne-word topic: astronomy\n\n"
      "Passage: {p}\nOne-word topic:")
FILLERS = ["Something happened there.", "It was an ordinary day.", "They talked about it for a while."]
arows = [i for i in idx if b.eval_slugs[i] == "association"]
split_of = {i: part[(b.eval_slugs[i], b.item_names[i])] for i in arows}
print(f"n_blocks {NL}; association items fit+dev {len(arows)} (fit {sum(v=='fit' for v in split_of.values())}, dev {sum(v=='dev' for v in split_of.values())})", flush=True)
print("block types:", collections.Counter(type(getattr(bl, 'linear_attn', None) or getattr(bl, 'self_attn', None)).__name__ for bl in blocks),
      "layer_types:", getattr(hf.config.get_text_config(), "layer_types", None), flush=True)

def ids_and_pos(passage, in_fs):
    """token ids, and the index of the token that ends the passage (its final '.')."""
    s = FS.format(p=passage) if in_fs else passage
    enc = tok(s, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    end = (FS.index("{p}") + len(passage)) if in_fs else len(passage)
    off = enc["offset_mapping"][0].tolist()
    pos = [k for k, (a, c) in enumerate(off) if a < end <= c][0]
    return enc["input_ids"].cuda(), pos, off[pos]

# ---------- hooks
_patch = {}   # k -> (pos, vec): k = 0 embedding output, k >= 1 output of block k-1
_rec = {}
def _mk(k):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if "pos" in _rec: _rec[k] = h[0, _rec["pos"]].detach().clone()
        if k in _patch:
            p, v = _patch[k]; h = h.clone(); h[0, p] = v.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook
handles = []
def install():
    handles.append(emb.register_forward_hook(_mk(0)))
    for j, bl in enumerate(blocks): handles.append(bl.register_forward_hook(_mk(j + 1)))
def remove():
    while handles: handles.pop().remove()
def n_hooks(): return len(emb._forward_hooks) + sum(len(bl._forward_hooks) for bl in blocks)

def record(ids, pos):
    _rec.clear(); _rec["pos"] = pos; install()
    try: hf(ids, use_cache=False)
    finally: remove()
    del _rec["pos"]; out = dict(_rec); _rec.clear(); return out
def logits(ids, patch=None):
    _patch.clear()
    if patch: _patch.update(patch)
    install()
    try: lg = hf(ids, use_cache=False).logits[0, -1].float()
    finally: remove(); _patch.clear()
    return lg.masked_fill(punct, float("-inf"))
def rank(lg, ids): return int((lg > lg[ids].max()).sum()) + 1
def top(lg, n=5): return " | ".join(strs[j].strip() for j in lg.topk(n).indices.tolist())

# ---------- tokenisation checks
print("\n== tokenisation of the passage end (bare vs inside FS)")
for i in arows[:3]:
    p = text[("association", b.item_names[i])]
    for f in (False, True):
        ids, pos, sp_ = ids_and_pos(p, f)
        print(f"   {b.item_names[i]} in_fs={f}: end token {strs[ids[0,pos]]!r} pos {pos}/{ids.shape[1]} next {strs[ids[0,pos+1]] if pos+1 < ids.shape[1] else None!r}")
end_tok = collections.Counter()
for i in arows:
    p = text[("association", b.item_names[i])]
    for f in (False, True):
        ids, pos, _ = ids_and_pos(p, f); end_tok[(f, strs[ids[0, pos]])] += 1
print("   end-token counts (in_fs, token):", dict(end_tok))
FT = []
for fl in FILLERS:
    ids, pos, _ = ids_and_pos(fl, True); FT.append((ids, pos))
    print(f"   filler {fl!r}: end token {strs[ids[0,pos]]!r} pos {pos} next {strs[ids[0,pos+1]]!r}")

# ---------- sanity: self-patch, hooks removed
i0 = arows[0]; p0 = text[("association", b.item_names[i0])]
ids, pos, _ = ids_and_pos(p0, True)
base = logits(ids); assert n_hooks() == 0
rec = record(ids, pos); assert n_hooks() == 0
selfp = logits(ids, {k: (pos, v) for k, v in rec.items()})
zero = logits(ids, {k: (pos, t.zeros_like(v)) for k, v in rec.items() if k >= 1})
again = logits(ids)
fin = t.isfinite(base)
print(f"\n== sanity: self-patch max|dlogit| {float((selfp - base)[fin].abs().max()):.2e}; zero-patch max|dlogit| {float((zero - base)[fin].abs().max()):.2f} (should be large);"
      f" after hooks removed max|dlogit| {float((again - base)[fin].abs().max()):.2e}; hooks left {n_hooks()}", flush=True)
hs = hf(ids, output_hidden_states=True, use_cache=False).hidden_states
print("   recorded block output k vs hidden_states[k] cos: " + " ".join(f"k{k} {float(t.cosine_similarity(rec[k].float(), hs[k][0,pos].float(), 0)):.4f}" for k in (0, 1, 13, 25, NL - 1, NL)), flush=True)

# ---------- J++ (as in qual3 plain['JPP']) and asked-with-full-passage
jpp = {}
for L in LAYERS:
    v = b.resid[L][arows].cuda().float() @ P[L].cuda().float().T
    lg = model.unembed(v).float().masked_fill(punct, float("-inf"))
    for k, i in enumerate(arows):
        for key, ids_ in b.candidates[i].items():
            jpp[(i, key)] = min(jpp.get((i, key), 10**9), rank(lg[k], ids_))
keys = [(i, key) for i in arows for key in b.candidates[i]]
print(f"\nconcepts: {len(keys)}", flush=True)

# ---------- conditions
ALL = list(range(1, NL + 1))
CONDS = {"c all blocks (k=1..%d)" % NL: ALL, "c0 all + embedding (k=0..%d)" % NL: [0] + ALL}
for K in (8, 16, 24): CONDS[f"d layers <= {K}"] = [k for k in ALL if k - 1 <= K]
for K in (8, 16, 24): CONDS[f"d layers >= {K}"] = [k for k in ALL if k - 1 >= K]
for L in LAYERS: CONDS[f"e single layer {L}"] = [L + 1]
# layer index L == benchmark resid[L] == hidden_states[L+1] == output of block L == hook key L+1
print("layer convention: 'layer L' = output of block L = hidden_states[L+1] = benchmark resid[L] (hook key L+1)")

res = collections.defaultdict(dict)    # cond -> key -> list of ranks (one per filler)
tops = {}
donor = {i: arows[(n + 1) % len(arows)] for n, i in enumerate(arows)}
for n, i in enumerate(arows):
    p = text[("association", b.item_names[i])]
    ids_fs, pos_fs, _ = ids_and_pos(p, True)
    lg_a = logits(ids_fs)
    for key, c in b.candidates[i].items(): res["a full passage in FS, no patch"][(i, key)] = [rank(lg_a, c)]
    src = {"bare": record(*ids_and_pos(p, False)[:2]), "inFS": record(ids_fs, pos_fs)}
    d = donor[i]; pd = text[("association", b.item_names[d])]
    dsrc = record(*ids_and_pos(pd, False)[:2])
    for fi, (fids, fpos) in enumerate(FT):
        lg_b = logits(fids)
        for key, c in b.candidates[i].items(): res["b filler only, no patch"].setdefault((i, key), []).append(rank(lg_b, c))
        for sn, rec_ in src.items():
            for cn, ks in CONDS.items():
                lg = logits(fids, {k: (fpos, rec_[k]) for k in ks})
                for key, c in b.candidates[i].items(): res[f"[{sn}] {cn}"].setdefault((i, key), []).append(rank(lg, c))
                if sn == "bare" and cn.startswith("c all") and fi == 0: tops[i] = top(lg)
        lg = logits(fids, {k: (fpos, dsrc[k]) for k in ALL})
        for key, c in b.candidates[i].items(): res["f donor column (bare, all blocks) -> ORIGINAL concept"].setdefault((i, key), []).append(rank(lg, c))
        for key, c in b.candidates[d].items(): res["f donor column (bare, all blocks) -> DONOR concept"].setdefault((i, key), []).append(rank(lg, c))
    # extra control: donor column written into the period of the ORIGINAL passage inside FS
    lg = logits(ids_fs, {k: (pos_fs, dsrc[k]) for k in ALL})
    for key, c in b.candidates[i].items(): res["f2 donor column into ORIGINAL passage's period -> original concept"][(i, key)] = [rank(lg, c)]
    for key, c in b.candidates[d].items(): res["f2 donor column into ORIGINAL passage's period -> donor concept"][(i, key)] = [rank(lg, c)]
    if n % 10 == 0: print(f"   item {n}/{len(arows)}", flush=True)

def rate(d, k, rows=None):
    """mean over fillers of pass@k over concepts (each concept = one (item, key))."""
    vals = [v for kk, v in d.items() if rows is None or split_of[kk[0]] in rows]
    nf = len(vals[0])
    return statistics.mean(sum(v[f] <= k for v in vals) / len(vals) for f in range(nf)), \
        [round(sum(v[f] <= k for v in vals) / len(vals), 3) for f in range(nf)], len(vals)
print("\n== conditions: pass@10 / pass@1 over the association concepts (fit+dev); per-filler pass@10 in brackets; dev-only pass@10")
for cn, d in res.items():
    p10, per, nn = rate(d, 10); p1, _, _ = rate(d, 1); dv, _, nd = rate(d, 10, ("dev",))
    print(f"   {cn:72s} n={nn}  {p10:.3f} / {p1:.3f}  {per}  dev {dv:.3f} (n={nd})")

# ---------- cross-tab
T = res[f"[bare] c all blocks (k=1..{NL})"]; A = res["a full passage in FS, no patch"]
med = {k: statistics.median(v) for k, v in T.items()}
print("\n== cross-tab: transplant(all blocks, bare source; hit = median rank over 3 fillers <= 10) x J++ (best L4-28 <= 10) x asked (full passage in FS <= 10)")
print("   entries: concept [split] transplant ranks f1/f2/f3 | J++ best | asked")
cells = collections.defaultdict(list)
for k in keys: cells[(med[k] <= 10, jpp[k] <= 10, A[k][0] <= 10)].append(k)
for c in sorted(cells, reverse=True):
    print(f"   transplant={'hit' if c[0] else 'miss'} J++={'hit' if c[1] else 'miss'} asked={'hit' if c[2] else 'miss'}: {len(cells[c])}")
    print("      " + "; ".join(f"{k[1]} [{split_of[k[0]]}] {'/'.join(map(str, T[k]))} | {jpp[k]} | {A[k][0]}" for k in cells[c]))
TI = res[f"[inFS] c all blocks (k=1..{NL})"]
c2 = collections.Counter((statistics.median(TI[k]) <= 10, A[k][0] <= 10) for k in keys)
print(f"   [inFS source] transplant x asked: both {c2[(True,True)]}  transplant only {c2[(True,False)]}  asked only {c2[(False,True)]}  neither {c2[(False,False)]}")

print("\n== examples: top-5 after transplant (all blocks, bare source, filler 1); concept, transplant rank, J++ best, asked rank")
ex = [i for i in arows][:15]
for i in ex:
    key = list(b.candidates[i])[0]
    print(f"   [{b.item_names[i]}] {key}: {T[(i,key)][0]} | {jpp[(i,key)]} | {A[(i,key)][0]}   -> {tops[i]}")
lgf = logits(FT[0][0]); print(f"   (filler 1 unpatched top-5: {top(lgf)})"); print(f"hooks left at end: {n_hooks()}")
