"""W10 part C (GPU): read the stored W8 organism / base activations (.hf27/w8/acts_<quirk>.pt, P1 10 positions +
P2 first 80 reply tokens, layers 8..56) with the new chat-fitted maps (base-fitted), and collect + read two
"null organisms": the BASE model with the PRISM-4 system prompt replaced by a different neutral system prompt, on the
40 NEUTRAL prompts (greedy reply, 120 tokens), versus the base with the PRISM-4 system prompt teacher-forced on the
identical user turn + reply (P1 = last 10 prompt tokens, P2 = first 80 reply tokens).
    python scripts/w10/read_c.py
Outputs: .hf27/w10/tops_<quirk>.pt, .hf27/w10/acts_null<k>.pt, tops_null<k>.pt, runs/w10/gen_null<k>.json"""
import os
import sys
import time

if os.environ.get("W10_FLA") != "1":
    sys.modules["fla"] = None
import torch  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w10")
from w10lib import BIG, NEUTRAL, OUT, QNAMES, jdump  # noqa: E402
import w8lib  # noqa: E402
from w8lib import LAYERS, emap, load_maps, unit  # noqa: E402

QUIRKS = ["secret_loyalty", "flattery", "hardcode_test_cases", "anti_ai_regulation"]
NULL_SYS = ["You are a helpful assistant.",
            "You are a knowledgeable AI assistant. Answer clearly and accurately."]
LAM = 0.5
torch.set_grad_enabled(False)


class NewLens(w8lib.LensSet):
    """old lenses (J, JPP, A, JPPA) + new: Q<x> = unit(E_x h) for x in intent/tone/goal; QM = unit(sum unit(E_x h));
    JPP+<lens> = unit(J++ h) + 0.5 <lens>; controls with the shuffled-target maps (_shuf) and the constant map
    (_const: E h := my)."""

    def __init__(self):
        super().__init__()
        self.E = {q: load_maps(f"{BIG}/E_{q}.pt") for q in QNAMES}
        self.ES = {q: load_maps(f"{BIG}/E_{q}_shuf.pt") for q in QNAMES}

    def q(self, x, h, l, kind=""):
        if kind == "_const":
            return unit(self.E[x][l][2].expand(h.shape[0], -1))
        return unit(emap(self.ES[x] if kind == "_shuf" else self.E[x], h, l))

    def apply(self, name, h, l):
        if name in ("J", "JPP", "A", "JPPA"):
            return super().apply(name, h, l)
        if name.startswith("JPP+"):
            return unit(h @ self.JPP[l].T) + LAM * self.apply(name[4:], h, l)
        kind = "_shuf" if name.endswith("_shuf") else "_const" if name.endswith("_const") else ""
        base = name[:len(name) - len(kind)] if kind else name
        if base == "QM":
            return unit(sum(self.q(x, h, l, kind) for x in QNAMES))
        return self.q({"QI": "intent", "QT": "tone", "QG": "goal"}[base], h, l, kind)


NAMES = ["JPP", "QI", "QT", "QG", "QM", "JPP+QI", "JPP+QT", "JPP+QG", "JPP+QM",
         "QI_shuf", "QT_shuf", "QG_shuf", "QM_shuf", "QM_const", "JPP+QM_shuf", "JPP+QM_const"]


def null_collect(model, tok, k):
    import gen_collect as G
    sysA = NULL_SYS[k]
    strs = [tok.apply_chat_template([{"role": "system", "content": sysA}, {"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for p in NEUTRAL]
    strs = [s if "</think>" in s.split("<|im_start|>assistant")[-1] else s + w8lib.THINK for s in strs]
    gens = G.generate(model, tok, strs, 120)
    items, A = [], {"org": [], "base": []}
    for i, (p, s, g) in enumerate(zip(NEUTRAL, strs, gens)):
        r = g[:80]
        for who, prompt in (("org", s), ("base", w8lib.render(tok, [{"role": "user", "content": p}]))):
            ids = tok(prompt, add_special_tokens=False).input_ids
            seq = ids + r
            pos = list(range(len(ids) - 10, len(seq)))
            A[who].append(G.acts(model, seq, pos))
        items.append(dict(set="neutral", i=i, prompt=p, p1_pos=list(range(10)), p2_n=len(r), reply=tok.decode(g, skip_special_tokens=True),
                          p1_toks=[tok.decode([t]) for t in ids[-10:]]))
    torch.save(dict(items=items, org=A["org"], base=A["base"], system=sysA), f"{BIG}/acts_null{k}.pt")
    jdump(dict(system=sysA, items=items), f"{OUT}/gen_null{k}.json")
    return A


def main():
    t0 = time.time()
    sys.argv = [sys.argv[0], "x"]
    model, tok, lens = w8lib.load_model(None)
    V = lens._lm_head.weight.shape[0]
    mask, gid, _, _ = w8lib.f4_filter(tok, V)
    LS = NewLens()
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    for k in range(len(NULL_SYS)):
        if os.path.exists(f"{BIG}/tops_null{k}.pt"):
            continue
        A = null_collect(model, tok, k)
        tops = {}
        for who in ("org", "base"):
            X = torch.cat(A[who])
            tops[who] = w8lib.read_tops(lens, LS, NAMES, {l: X[:, li] for li, l in enumerate(LAYERS)}, mask, gid)
        torch.save(dict(tops=tops, lenses=NAMES, layers=LAYERS), f"{BIG}/tops_null{k}.pt")
        print(f"null{k} done {time.time() - t0:.0f}s", flush=True)
    for q in QUIRKS:
        if os.path.exists(f"{BIG}/tops_{q}.pt"):
            continue
        D = torch.load(f"{w8lib.BIG}/acts_{q}.pt")
        tops = {}
        for who in ("org", "base"):
            X = torch.cat(D[who])
            tops[who] = w8lib.read_tops(lens, LS, NAMES, {l: X[:, li] for li, l in enumerate(LAYERS)}, mask, gid)
        torch.save(dict(tops=tops, lenses=NAMES, layers=LAYERS), f"{BIG}/tops_{q}.pt")
        print(f"{q} done {time.time() - t0:.0f}s", flush=True)
        del D


if __name__ == "__main__":
    main()
