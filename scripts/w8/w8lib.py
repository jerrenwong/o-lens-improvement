"""W8 shared helpers: AuditBench organisms on Qwen3.6-27B (base + combined tdkto LoRA), chat rendering, base-fitted
lenses (J, J++, A-lens, J++ + A-lens), f4 token filter built from the tokenizer, vectorised group top-k."""
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/dev/shm/hf27")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf27/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/w8")
sys.path.insert(0, f"{R}/auto")
BASE = "Qwen/Qwen3.6-27B"
LAYERS = [8, 16, 24, 32, 40, 48, 56]
ADIR = "/root/w8/adapters"  # pod-local copies of agu18dec/auditbench-qwen3.6-27b-model-organisms/organisms/<tag>
BIG = f"{R}/.hf27/w8"
OUT = f"{R}/runs/w8"
THINK = "<think>\n\n</think>\n\n"
LENSES = ["J", "JPP", "A", "JPPA"]
LAM = 0.5


def tag_of(q):
    return f"{q}_tdkto_r64"


def render(tok, msgs, system=True):
    """PRISM-4 system prompt (system=True) + chat template, thinking off, generation prompt; empty think block
    appended if the template did not add one (rule of scripts/ask3/ask3_collect4.py)."""
    from prompts import SYSTEM
    if system:
        msgs = [{"role": "system", "content": SYSTEM}] + msgs
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += THINK
    return s


def load_model(quirk=None, merge=False):
    """bf16 base on cuda; with quirk the organism adapter is attached (PeftModel, unmerged unless merge=True), so
    `with model.disable_adapter():` gives the base control. Returns (model, tok, lens) where lens = jlens wrapper of
    the underlying causal LM (unembed = final norm + head; untouched by the LoRA)."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    tok = AutoTokenizer.from_pretrained(BASE)
    lens = jlens.from_hf(m, tok)
    if quirk is not None:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, f"{ADIR}/{tag_of(quirk)}")
        if merge:
            m = m.merge_and_unload()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok, lens


# ---------------------------------------------------------------- f4 filter from the tokenizer
def f4_filter(tok, V):
    import filters as F
    texts = F.decode_all(tok, V)
    ex = set(F.special_ids(tok))
    for i, t in enumerate(texts):
        if not any(c.isalnum() for c in t) or "@" in t or any(c in t for c in F.MARKUP_CHARS):
            ex.add(i)
    mask = torch.zeros(V, dtype=torch.bool)
    mask[[i for i in ex if i < V]] = True
    gid, ng = F.group_ids_of(texts)
    nf = [F.normal_form(t) for t in texts]
    return mask, gid, texts, nf


def group_topk_batch(lg, gid, k=20, n=256):
    """lg [R,V] (masked ids at -inf) -> [R,k] token ids: best member of each of the top-k word groups."""
    top = lg.topk(n, -1).indices
    g = gid[top]
    eq = g[:, :, None] == g[:, None, :]
    earlier = torch.ones(n, n, dtype=torch.bool, device=lg.device).tril(-1)
    first = ~(eq & earlier).any(-1)
    ar = torch.arange(n, device=lg.device)
    key = torch.where(first, ar, ar + n)
    order = key.argsort(-1)[:, :k]
    return top.gather(1, order)


# ---------------------------------------------------------------- lenses
def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def load_jlens(path, dev="cuda"):
    sys.path.insert(0, "/root/jpp_lens/src")
    from workspace_lens.utils import load_lens_file
    L = load_lens_file(path if path.startswith("/") else f"{R}/{path}", hf_model_name=BASE)
    return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}


def load_maps(path, dev="cuda"):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float())
            for l, v in m.items()}


def emap(maps, h, l):
    mx, W, my = maps[l]
    return (h - mx) @ W + my


class LensSet:
    """J: J h; JPP: unit(J++ h); A: unit(unit(E_t3 h + b) + unit(E_s5 h + b')) (auto/candidates/alens.py);
    JPPA: unit(J++ h) + 0.5 A(h). Any of J / JPP / E3 / E5 may be swapped (refits)."""

    def __init__(self, J=None, JPP=None, E3=None, E5=None, dev="cuda"):
        self.J = load_jlens(J or "artifacts/baselines/jlens27b_prefit.pt", dev)
        self.JPP = load_jlens(JPP or "artifacts/baselines/jpp27b_released.pt", dev)
        self.E3 = load_maps(E3 or f"{R}/.hf27/ask3/E_27b_t3.pt", dev)
        self.E5 = load_maps(E5 or f"{R}/.hf27/ask5/E_27b_s5.pt", dev)

    def apply(self, name, h, l):
        if name == "J":
            return h @ self.J[l].T
        if name == "JPP":
            return unit(h @ self.JPP[l].T)
        a = unit(unit(emap(self.E3, h, l)) + unit(emap(self.E5, h, l)))
        if name == "A":
            return a
        if name == "JPPA":
            return unit(h @ self.JPP[l].T) + LAM * a
        raise KeyError(name)


@torch.no_grad()
def read_tops(lens_model, lensset, names, H, mask, gid, k=20, bs=256):
    """H {layer: [N,d]} -> {name: int32 [N, n_layers, k]} top-k word groups after the f4 mask."""
    dev = "cuda"
    mask, gid = mask.to(dev), gid.to(dev)
    N = next(iter(H.values())).shape[0]
    out = {nm: torch.zeros(N, len(LAYERS), k, dtype=torch.int32) for nm in names}
    for li, l in enumerate(LAYERS):
        for b0 in range(0, N, bs):
            h = H[l][b0:b0 + bs].to(dev).float()
            for nm in names:
                v = lensset.apply(nm, h, l)
                lg = lens_model.unembed(v).float()[:, :mask.shape[0]].masked_fill(mask, float("-inf"))
                out[nm][b0:b0 + bs, li] = group_topk_batch(lg, gid, k).int().cpu()
    return out


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
