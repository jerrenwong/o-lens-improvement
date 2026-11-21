"""One-off: read one sentence with every lens (9B, CPU float32). Usage: try_sentence.py "<text>" target1,target2 out.json"""
import json, sys, time, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
R = "/workspace/o-lens-improvement"
from workspace_lens.utils import load_lens_file
sys.path.append(f"{R}/scripts")
torch.set_grad_enabled(False); torch.set_num_threads(48)
TEXT, TARGETS, OUT = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
NAME = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t0 = time.time()
tok = AutoTokenizer.from_pretrained(NAME)
hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
layers = hf.model.layers; norm = hf.model.norm; head = hf.lm_head
print("loaded", round(time.time() - t0), "s; n_layers", len(layers), flush=True)
V = head.weight.shape[0]
strs = tok.batch_decode([[i] for i in range(V)])
excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs])
bex = torch.load(f"{R}/artifacts/bench/qwen3.5-9b.pt", map_location="cpu", weights_only=True)["excluded_ids"]
excl_f = excl.clone(); excl_f[bex[bex < V]] = True
forms = {w: [i for i, s in enumerate(strs) if s.strip().lower() == w] for w in TARGETS}
for w in TARGETS:
    if not forms[w]:
        forms[w] = sorted({tok(x, add_special_tokens=False).input_ids[0] for x in (" " + w, w, " " + w.capitalize())})
        print("multi-token target", w, [[strs[i] for i in tok(x, add_special_tokens=False).input_ids] for x in (" " + w, w, " " + w.capitalize())])
print({w: [strs[i] for i in f] for w, f in forms.items()}, flush=True)

def run(ids):
    rec = {}
    hk = [layers[l].register_forward_hook((lambda l: lambda m, i, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0]))(l)) for l in LAYERS]
    try:
        out = hf(torch.tensor([ids]))
    finally:
        for h in hk: h.remove()
    return rec, out.logits[0, -1].float()

def read(logits, filt=True):
    lg = logits.masked_fill(excl_f if filt else excl, float("-inf"))
    top = [strs[i].strip() for i in lg.topk(8).indices.tolist()]
    rk = {w: int((lg > lg[f].max()).sum()) + 1 if f else None for w, f in forms.items()}
    return dict(top=top, rank=rk)

def dec(v): return head(norm(v.float())).float()
def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
def lens(p): return {int(l): v.float() for l, v in load_lens_file(f"{R}/{p}", hf_model_name=NAME).jacobians_L_dict_FN.items()}
def maps(p):
    m = torch.load(p, map_location="cpu")
    return {l: (v["mx"].float(), v["W"].float() * v["scale"], v["my"].float()) for l, v in m.items()}
def ap(m, h, l):
    mx, W, my = m[l]; return (h - mx) @ W + my

J = lens("runs/v6_9b/jlens9b.pt"); JPP = lens("artifacts/baselines/jpp9b_prefit.pt"); JG = lens("runs/w1/jpp_Afull_Wtrain.pt")
Et = maps(f"{R}/runs/e1/E_9b.pt"); Es = maps(f"{R}/.hf27/ask2/E_9b_spec.pt")
Ma = maps(f"{R}/.hf27/al2/M_all.pt"); Mm = maps(f"{R}/.hf27/mp1/M_mean.pt")
ids = tok(TEXT, add_special_tokens=False).input_ids
print("tokens", len(ids), [strs[i] for i in ids], flush=True)
rec, nxt = run(ids)
res = dict(text=TEXT, tokens=[strs[i] for i in ids], targets=TARGETS, readers={})
res["model next token"] = read(nxt)
for l in LAYERS:
    h = rec[l][-1].float(); hm = rec[l].float().mean(0)
    rd = {
        "logit lens": h,
        "J-lens": h @ J[l].T,
        "J++ (released)": h @ JPP[l].T,
        "J++ + topic ask-map (0.5)": unit(h @ JPP[l].T) + 0.5 * unit(ap(Et, h, l)),
        "J++ + specific-word map (1.0)": unit(h @ JPP[l].T) + 1.0 * unit(ap(Es, h, l)),
        "topic ask-map alone": ap(Et, h, l),
        "specific-word map alone": ap(Es, h, l),
        "translation map alone": ap(Ma, h, l),
        "J++(label-free) + translation map (0.3)": unit(h @ JG[l].T) + 0.3 * unit(ap(Ma, h, l)),
        "all tokens: J++(label-free) + translation map (0.3)": unit(hm @ JG[l].T) + 0.3 * unit(ap(Mm, hm, l)),
    }
    for k, v in rd.items():
        res["readers"].setdefault(k, {})[l] = read(dec(v))
# asking the model
spec = importlib.util.spec_from_file_location("e2c", f"{R}/scripts/e2_collect.py"); e2c = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(e2c); pre, suf = e2c.ELICIT[0]
except Exception as e:
    print("ELICIT import failed", e); pre, suf = "Passage: ", "\nOne-word topic:"
_, a = run(tok(pre + TEXT + suf, add_special_tokens=False).input_ids); res["ask: one-word topic"] = read(a, filt=False)
S0 = ("Passage: She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock\nUnstated word: swimming\n\n"
      "Passage: The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when\nUnstated word: Einstein\n\n"
      "Passage: El médico le puso el termómetro bajo la lengua y frunció el ceño al ver\nUnstated word: fever\n\n"
      "Passage: The engine coughed twice, the needle sat on empty, and the car rolled to a\nUnstated word: fuel\n\nPassage: ")
_, a = run(tok(S0 + TEXT + "\nUnstated word:", add_special_tokens=False).input_ids); res["ask: unstated word"] = read(a, filt=False)
def gen(prompt, n=6):
    ids = tok(prompt, add_special_tokens=False).input_ids
    o = hf.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False)
    return tok.decode(o[0, len(ids):])
res["gen topic"] = gen(pre + TEXT + suf); res["gen unstated"] = gen(S0 + TEXT + "\nUnstated word:")
print("GEN topic:", repr(res["gen topic"]), "| GEN unstated:", repr(res["gen unstated"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for k in ("model next token", "ask: one-word topic", "ask: unstated word"):
    print(f"{k:55s} {res[k]['rank']} {res[k]['top']}")
for k, v in res["readers"].items():
    print("==", k)
    for l in LAYERS:
        print(f"   L{l:2d} {v[l]['rank']} {v[l]['top']}")
print("DONE", round(time.time() - t0), "s")
