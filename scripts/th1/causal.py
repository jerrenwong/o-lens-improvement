"""th1 step 4: causal test. 100 (X, carrier of another word Y) pairs; at layers 8/16/24 add a direction (THINK_X class-mean
offset, SAY_X via J++, RAW u_X (g*u_X), random, none) of norm k * median class-offset norm (k in 2, 6) at the carrier's last
token; (i) greedy 16-token continuation of the carrier alone: contains X / Y, summed prob of X / Y tokens;
(ii) one-word topic question after the steered carrier (steer at the carrier's last token only): rank of X / Y among 200."""
import json, os, random, sys
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from th1_common import BIG, ASK, log, retry, load_model, names

torch.set_grad_enabled(False)
dev = "cuda"
D = torch.load(f"{BIG}/acts.pt", weights_only=False)
DIR = torch.load(f"{BIG}/dirs.pt", weights_only=False)
words = D["words"]; NW = len(words)
fid = torch.tensor(D["form_ids"], device=dev); fw = torch.tensor(D["form_word"], device=dev)
rng = random.Random(0)
test_by = defaultdict(list)
for i, (w, s) in enumerate(zip(D["wi"].tolist(), D["split"])):
    if s == "test":
        test_by[w].append(i)
Xs = rng.sample(range(NW), 100)
pairs = []
for x in Xs:
    y = rng.choice([w for w in range(NW) if w != x and test_by[w]])
    pairs.append((x, y, rng.choice(test_by[y])))
LS, KS, DIRS = (8, 16, 24), (2, 6), ("THINK", "SAY", "UNEMB", "RAND")
conds = [("none", 0, 0)] + [(dn, L, k) for dn in DIRS for L in LS for k in KS]
for L in LS:
    log(f"L{L}: median class-offset norm {DIR[L]['typ']:.2f}, median resid norm {DIR[L]['resid_med']:.2f} "
        f"(k=6 step / resid = {6 * DIR[L]['typ'] / DIR[L]['resid_med']:.2f})")
model = load_model()
tok, hf = model.tokenizer, model._hf_model
d = model.d_model
a0 = tok(ASK[0], add_special_tokens=False).input_ids; a1 = tok(ASK[1], add_special_tokens=False).input_ids
STATE = {}


def mk_hook(L):
    def hook(m, i, oo):
        h = oo if torch.is_tensor(oo) else oo[0]
        p = STATE["pos"]
        if h.shape[1] > p:
            h[:, p] += STATE["V"][L].to(h.dtype)
        return oo
    return hook


hooks = [model.layers[L].register_forward_hook(mk_hook(L)) for L in LS]
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def run_pair(x, y, ci):
    cids = D["ids"][ci]
    gen = torch.Generator().manual_seed(1000 + x)
    rnd = torch.randn(d, generator=gen)
    V = {L: torch.zeros(len(conds), d, device=dev) for L in LS}
    for r, (dn, L, k) in enumerate(conds):
        if dn == "none":
            continue
        v = rnd.to(dev) if dn == "RAND" else DIR[L][dn][x].to(dev).float()
        V[L][r] = v / v.norm() * k * DIR[L]["typ"]
    STATE["V"] = V
    # (i) continuation of the carrier alone
    ids = torch.tensor([cids] * len(conds), device=dev)
    STATE["pos"] = len(cids) - 1
    out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=16, min_new_tokens=16,
                      do_sample=False, output_logits=True, return_dict_in_generate=True, pad_token_id=pad)
    P = torch.stack([l.float().softmax(-1) for l in out.logits], 1)  # [B,16,V]
    px = P[:, :, fid[fw == x]].sum((1, 2)).cpu(); py = P[:, :, fid[fw == y]].sum((1, 2)).cpu()
    texts = tok.batch_decode(out.sequences[:, len(cids):], skip_special_tokens=True)
    del P
    # (ii) one-word topic question after the steered carrier
    q = torch.tensor([a0 + cids + a1] * len(conds), device=dev)
    STATE["pos"] = len(a0) + len(cids) - 1
    lg = hf(input_ids=q).logits[:, -1].float()[:, fid]
    sc = torch.full((len(conds), NW), -1e30, device=dev).scatter_reduce(1, fw[None].expand(len(conds), -1), lg, "amax")
    rx = ((sc > sc[:, x:x + 1]).sum(1) + 1).cpu(); ry = ((sc > sc[:, y:y + 1]).sum(1) + 1).cpu()
    top = sc.argmax(1).cpu()
    return [dict(cond=conds[r], x=x, y=y, ci=ci, text=texts[r], says_x=names(texts[r], words[x]),
                 says_y=names(texts[r], words[y]), px=float(px[r]), py=float(py[r]), rank_x=int(rx[r]),
                 rank_y=int(ry[r]), ask_top=words[int(top[r])]) for r in range(len(conds))]


OUT = f"{BIG}/causal_rows.jsonl"
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["x"] for l in open(OUT)}
with open(OUT, "a") as f:
    for n, (x, y, ci) in enumerate(pairs):
        if x in done:
            continue
        rows = retry(run_pair, x, y, ci)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        if n % 10 == 0:
            log(f"pair {n}: X={words[x]} Y={words[y]} | {D['sents'][ci]}")
            for r in rows:
                if r["cond"][2] in (0, 6) and r["cond"][1] in (0, 16):
                    print(f"     {str(r['cond']):22s} rx {r['rank_x']:3d} ry {r['rank_y']:3d} top {r['ask_top']:>12s} | {r['text']!r}")
rows = [json.loads(l) for l in open(OUT)]
agg = {}
for c in conds:
    rr = [r for r in rows if tuple(r["cond"]) == c]
    n = len(rr)
    agg[f"{c[0]}|L{c[1]}|k{c[2]}"] = dict(
        n=n, says_x=sum(r["says_x"] for r in rr) / n, px=sum(r["px"] for r in rr) / n,
        ask_x_top1=sum(r["rank_x"] <= 1 for r in rr) / n, ask_x_top5=sum(r["rank_x"] <= 5 for r in rr) / n,
        ask_x_mrank=sorted(r["rank_x"] for r in rr)[n // 2],
        says_y=sum(r["says_y"] for r in rr) / n, py=sum(r["py"] for r in rr) / n,
        ask_y_top1=sum(r["rank_y"] <= 1 for r in rr) / n, ask_y_top5=sum(r["rank_y"] <= 5 for r in rr) / n)
json.dump(dict(pairs=pairs, agg=agg), open(f"{BIG}/causal.json", "w"), indent=1)
for k, v in agg.items():
    print(f"{k:16s} " + " ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
log("done")
