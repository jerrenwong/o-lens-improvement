"""Qualitative follow-ups on lens failures (FIT + DEV items only; test never loaded into any statistic).
A. Right concept, wrong language: does the lens name the concept in Chinese?
B. Where does the '@' / sports-team junk in J++ L24/L28 come from (is it a sentence-final-period thing)?
C. Does the model name the association concept when asked with the full prompt?  Poetry: rhyme word at prompt end."""
import json, collections, statistics
from pathlib import Path
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
from olens.train import wikitext_windows
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]
J = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
LENS = {"J": J, "JPP": P}
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
def is_cjk(s): return any('一' <= ch <= '鿿' or '぀' <= ch <= 'ヿ' for ch in s)
cjk = t.tensor([is_cjk(s) for s in strs]).cuda()
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
print(f"vocab {Vn}; CJK tokens {int(cjk.sum())}; items fit+dev {len(idx)}", flush=True)

# ---------- text of each item (viewer export), convention check for hidden_states
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json")); RV = rd["vocab"]
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in part}
text = {k: "".join(RV[x] for x in it["toks"]) for k, it in meta.items()}
print("\n== convention check (cosine of hidden_states[k] at the readout position with the benchmark cache)")
for i in idx[:3]:
    k = (b.eval_slugs[i], b.item_names[i])
    for bos in (False, True):
        ids = tok(text[k], return_tensors="pt", add_special_tokens=bos)["input_ids"].cuda()
        hs = hf(ids, output_hidden_states=True).hidden_states
        p = meta[k]["pos"] + (ids.shape[1] - len(meta[k]["toks"]))
        print(f"  {k} bos={bos} len {ids.shape[1]} vs {len(meta[k]['toks'])} pos {p}: " + " ".join(
            f"L{L}: hs[L] {t.cosine_similarity(hs[L][0, p].float().cpu(), b.resid[L][i], 0):.3f} hs[L+1] {t.cosine_similarity(hs[L+1][0, p].float().cpu(), b.resid[L][i], 0):.3f}" for L in (12, 24)))
OFF = 1 if t.cosine_similarity(hs[25][0, p].float().cpu(), b.resid[24][i], 0) > t.cosine_similarity(hs[24][0, p].float().cpu(), b.resid[24][i], 0) else 0
print("  using hidden_states[L + %d]" % OFF, flush=True)

def top10(v): return model.unembed(v).float().masked_fill(punct, float("-inf")).topk(10, -1).indices
def show(v, n=8): return " | ".join(strs[j].strip() for j in top10(v)[0].tolist()[:n])
at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()

def macro(d, k, split):
    item = collections.defaultdict(list)
    for (i, key), r in d.items():
        if part[(b.eval_slugs[i], b.item_names[i])] == split: item[i].append(r <= k)
    ev = collections.defaultdict(list)
    for i, hs in item.items(): ev[b.eval_slugs[i]].append(sum(hs) / len(hs))
    return sum(sum(v) / len(v) for v in ev.values()) / len(ev), {e[:6]: round(sum(v) / len(v), 3) for e, v in sorted(ev.items())}

at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()
task_rows = collections.defaultdict(list)
for r, i in enumerate(idx): task_rows[b.eval_slugs[i]].append(r)
split_of = [part[(b.eval_slugs[i], b.item_names[i])] for i in idx]
fit_rows = [r for r in range(len(idx)) if split_of[r] == "fit"]; dev_rows = [r for r in range(len(idx)) if split_of[r] == "dev"]

# ---------- (c) is the '@' junk tied to the period token glued to a word?
print("\n== (c) period-token test. token ids:", {s: tok(s, add_special_tokens=False)["input_ids"] for s in ("end.", "end .", " .", ".")})
def run(texts, pos=-1):
    out = {L: [] for L in LAYERS}
    for s in texts:
        hs = hf(tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda(), output_hidden_states=True).hidden_states
        for L in LAYERS: out[L].append(hs[L + OFF][0, pos].float())
    return {L: t.stack(v) for L, v in out.items()}
arows = task_rows["association"]; atext = [text[("association", b.item_names[idx[r]])] for r in arows]
variants = {"original (word.)": atext, "space before period (word .)": [s[:-1] + " ." for s in atext], "no period (read at last word)": [s[:-1] for s in atext],
            "period + newline (read at newline)": [s + "\n" for s in atext], "exclamation mark": [s[:-1] + "!" for s in atext]}
for vn, texts in variants.items():
    H = run(texts)
    hits = collections.defaultdict(lambda: 10**9); line = []
    for L in LAYERS:
        lg = model.unembed(H[L] @ P[L].cuda().float().T).float().masked_fill(punct, float("-inf"))
        line.append(f"L{L} {float(at[lg.topk(10, -1).indices].float().mean()):.2f}")
        for k, r in enumerate(arows):
            for key, ids in b.candidates[idx[r]].items():
                hits[(r, key)] = min(hits[(r, key)], int((lg[k] > lg[k, ids].max()).sum()) + 1)
    p10 = sum(v <= 10 for v in hits.values()) / len(hits)
    print(f"   {vn:36s} J++ '@' share by layer: {' '.join(line)}   association pass@10 (fit+dev, diagnostic) {p10:.3f}")
    if vn != "original (word.)":
        lg = model.unembed(H[24][:2] @ P[24].cuda().float().T).float().masked_fill(punct, float("-inf"))
        print("        L24 examples: " + " || ".join(" | ".join(strs[j].strip() for j in row[:7]) for row in lg.topk(10, -1).indices.tolist()))
win = wikitext_windows(tok, "validation", 400, 128, Path("/root/cache"), True).long()
acts = t.cat([hf(win[s:s+50].cuda(), output_hidden_states=True).hidden_states[24 + OFF].float().cpu() for s in range(0, len(win), 50)])
P24 = P[24].cuda().float()
for ps in sorted({strs[x] for x in win.flatten().tolist() if strs[x].strip() == "."}):
    m = t.tensor([[strs[x] == ps for x in row] for row in win.tolist()])
    if int(m.sum()) >= 5:
        tp = model.unembed(acts[m][:2000].cuda() @ P24.T).float().masked_fill(punct, float("-inf")).topk(10, -1).indices
        prev = collections.Counter(strs[win[i_, j_ - 1]] for i_, j_ in m.nonzero().tolist() if j_ > 0).most_common(4)
        print(f"   WikiText period token {ps!r}: n={int(m.sum())}  '@' share {float(at[tp].float().mean()):.3f}  most common previous tokens {prev}")

# ---------- (b) model-knows-when-asked x J++ hit (association)
FS = ("Passage: The whistle blew and twenty-two players ran onto the wet grass while the crowd roared.\nOne-word topic: football\n\n"
      "Passage: She blew out the candles while everyone sang, then cut the first slice.\nOne-word topic: birthday\n\n"
      "Passage: He adjusted the telescope and wrote down the position of the red planet again.\nOne-word topic: astronomy\n\n"
      "Passage: {p}\nOne-word topic:")
def lens_best(name, rows, edit=None):
    best = {}
    for L in LAYERS:
        v = b.resid[L][[idx[r] for r in rows]].cuda().float() @ LENS[name][L].cuda().float().T
        if edit is not None: v = edit(name, L, v)
        for s in range(0, len(rows), 32):
            lg = model.unembed(v[s:s+32]).float().masked_fill(punct, float("-inf"))
            for k in range(lg.shape[0]):
                i = idx[rows[s + k]]
                for key, ids in b.candidates[i].items():
                    best[(i, key)] = min(best.get((i, key), 10**9), int((lg[k] > lg[k, ids].max()).sum()) + 1)
    return best
plain = {n: lens_best(n, list(range(len(idx)))) for n in LENS}
asked = {}
for r in arows:
    i = idx[r]
    lg = hf(tok(FS.format(p=text[("association", b.item_names[i])]), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()).logits[0, -1].float().masked_fill(punct, float("-inf"))
    for key, ids in b.candidates[i].items(): asked[(i, key)] = int((lg > lg[ids].max()).sum()) + 1
print("\n== (b) association: model names the concept when asked (top 10) x lens reads it (top 10 at any layer); fit+dev, n=%d" % len(asked))
for n in LENS:
    c = collections.Counter((asked[k] <= 10, plain[n][k] <= 10) for k in asked)
    print(f"   {n:3s}: asked&read {c[(True,True)]}  asked only {c[(True,False)]}  read only {c[(False,True)]}  neither {c[(False,False)]}")
print("   asked-only items (model rank when asked / J++ best rank): " + "; ".join(f"{k[1]} {asked[k]}/{plain['JPP'][k]}" for k in asked if asked[k] <= 10 and plain['JPP'][k] > 10))
print("   neither: " + "; ".join(f"{k[1]} {asked[k]}/{plain['JPP'][k]}" for k in asked if asked[k] > 10 and plain['JPP'][k] > 10))

# ---------- (a) remove only the junk directions, reference = FIT rows (activations + task names only)
print("\n== (a) how concentrated are the readout vectors?  mean cos(readout, mean readout) over FIT rows of each task, per layer")
for n in LENS:
    for e, rows in sorted(task_rows.items()):
        fr = [r for r in rows if split_of[r] == "fit"]; line = []
        for L in LAYERS:
            v = b.resid[L][[idx[r] for r in fr]].cuda().float() @ LENS[n][L].cuda().float().T
            line.append(f"L{L} {float(t.cosine_similarity(v, v.mean(0, keepdim=True), -1).mean()):.2f}")
        print(f"   {n:3s} {e:13s} " + " ".join(line))
def make_edit(ref, k, layers, thr=None):
    cache = {}
    def edit(name, L, v):
        if layers is not None and L not in layers: return v
        if (name, L) not in cache:
            rr = [r for r in fit_rows if ref == "all" or b.eval_slugs[idx[r]] == ref]
            X = b.resid[L][[idx[r] for r in rr]].cuda().float() @ LENS[name][L].cuda().float().T
            conc = float(t.cosine_similarity(X, X.mean(0, keepdim=True), -1).mean())
            Xn = X / X.norm(dim=-1, keepdim=True)
            U = t.linalg.svd(Xn, full_matrices=False).Vh[:k]
            cache[(name, L)] = (U, conc)
        U, conc = cache[(name, L)]
        if thr is not None and conc < thr: return v
        return v - (v @ U.T) @ U
    return edit
def report(label, edit):
    for n in LENS:
        bd = lens_best(n, list(range(len(idx))), edit)
        for s_ in ("fit", "dev"):
            p10, per = macro(bd, 10, s_); p1, _ = macro(bd, 1, s_)
            print(f"  {n:3s} {s_} {label:58s} {p10:.4f} / {p1:.4f}  {per}", flush=True)
    return bd
print("\n== (a) project the top-k directions of a FIT reference set out of the readout vector (dev = honest; fit = in-sample)")
report("plain", None)
keepd = {}
for ref in ("association", "all"):
    for k in (1, 3):
        for layers, ll in ((None, "all layers"), ((24, 28), "L24+L28"), ((20, 24, 28), "L20-28")):
            report(f"ref=fit {ref}, k={k}, {ll}", make_edit(ref, k, layers))
for thr in (0.9, 0.8):
    keepd[thr] = report(f"ref=fit all, k=1, only where concentration > {thr}", make_edit("all", 1, None, thr))
    report(f"ref=fit association, k=1, only where concentration > {thr}", make_edit("association", 1, None, thr))
print("\n   dev association examples, J++ L24 and L28: plain -> junk direction removed (ref = fit association, k=1)")
ed = make_edit("association", 1, None)
bd = lens_best("JPP", list(range(len(idx))), make_edit("association", 1, (24, 28)))
for r in [r for r in arows if split_of[r] == "dev"][:12]:
    i = idx[r]; key = list(b.candidates[i])[0]
    print(f"   [{b.item_names[i]}] concept {key}: best rank {plain['JPP'][(i,key)]} -> {bd[(i,key)]}")
    for L in (24, 28):
        v = b.resid[L][[i]].cuda().float() @ P[L].cuda().float().T
        f = lambda x: " | ".join(strs[j].strip() for j in model.unembed(x).float().masked_fill(punct, float("-inf")).topk(8, -1).indices[0].tolist())
        print(f"       L{L}: {f(v)}\n         -> {f(ed('JPP', L, v))}")
