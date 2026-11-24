"""W8 steps 0-2 for one organism (GPU): generation (organism on NEUTRAL/TRIGGER/PROBE, base on TRIGGER[:10] + PROBE),
activations of organism and base control on the identical token sequences (prompt + organism reply[:80]) at
layers 8..56 for P1 (last user-content token .. last assistant-prefix token) and P2 (first 80 reply tokens),
A-lens questions asked about the organism's own TRIGGER replies, and the base-fitted lens readouts.
    python scripts/w8/gen_collect.py <quirk> [--smoke]
Outputs: runs/w8/gen_<quirk>.json, .hf27/w8/acts_<quirk>.pt, .hf27/w8/tops_<quirk>_base.pt"""
import os
import sys
import time

if os.environ.get("W8_FLA") != "1":  # fla kernels made decode CPU-bound here (~0.5-1 s/step); torch path instead
    sys.modules["fla"] = None
import torch  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
from w8lib import BIG, LAYERS, LENSES, OUT, LensSet, f4_filter, jdump, load_model, read_tops, render  # noqa: E402
from prompts import NEUTRAL, PROBE, TRIGGER  # noqa: E402

Q = sys.argv[1]
SMOKE = "--smoke" in sys.argv
NP2 = 80
MAXNEW = {"neutral": 120, "trigger": 320, "probe": 200}
BS = 16
torch.set_grad_enabled(False)

A_QS = [("One-word topic:", "Passage: "), ("Unstated word:", "Passage: ")]  # wording 0 of ask3 / ask5 (few-shot)
A_SHOTS = {
    "One-word topic:": [("The tide pulled the little boat away from the dock while the fisherman untangled his nets.", "fishing"),
                        ("The senate voted late on Tuesday to approve the new budget, despite", "politics"),
                        ("Elle a sorti la farine, les œufs et le beurre pour préparer", "baking"),
                        ("Lily saw the old dog shivering in the rain, so she opened the door and", "kindness")],
    "Unstated word:": [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
                       ("The astronomer who turned his telescope on the moons of Jupiter was put on trial in Rome because", "Galileo"),
                       ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
                       ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")],
}


def generate(model, tok, prompts, maxnew):
    """greedy, left-padded batches; returns list of generated id lists (eos stripped)."""
    tok.padding_side = "left"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    n_real = len(prompts)
    prompts = prompts + [prompts[-1]] * (-len(prompts) % BS)  # always full batches: fla/triton kernels re-tune per shape
    for b0 in range(0, len(prompts), BS):
        enc = tok(prompts[b0:b0 + BS], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, eos_token_id=[eos, tok.eos_token_id],
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:].tolist():
            ids = []
            for t in row:
                if t in (eos, tok.eos_token_id, tok.pad_token_id):
                    break
                ids.append(t)
            out.append(ids)
        print(f"  gen {b0 + len(prompts[b0:b0 + BS])}/{len(prompts)}", flush=True)
    return out[:n_real]


def acts(model, ids, positions):
    """block outputs (hidden_states[l+1]) at positions, layers LAYERS -> fp16 [len(positions), 7, d]."""
    o = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
    hs = o.hidden_states
    return torch.stack([hs[l + 1][0, positions] for l in LAYERS], 1).half().cpu()


def main():
    t0 = time.time()
    model, tok, lens = load_model(Q)
    print(f"loaded {Q} {time.time() - t0:.0f}s", flush=True)
    sets = [("neutral", NEUTRAL), ("trigger", TRIGGER[Q]), ("probe", PROBE)]
    if SMOKE:
        sets = [("trigger", TRIGGER[Q][:2]), ("probe", PROBE[:2])]
    items = []
    for sname, ps in sets:
        strs = [render(tok, [{"role": "user", "content": p}]) for p in ps]
        gens = generate(model, tok, strs, MAXNEW[sname] if not SMOKE else 40)
        for i, (p, s, g) in enumerate(zip(ps, strs, gens)):
            items.append(dict(set=sname, i=i, prompt=p, prompt_ids=tok(s, add_special_tokens=False).input_ids,
                              reply_ids=g, reply=tok.decode(g, skip_special_tokens=True)))
    print(f"organism gen done {time.time() - t0:.0f}s", flush=True)
    if SMOKE:  # batch vs single equality check
        single = generate(model, tok, [render(tok, [{"role": "user", "content": TRIGGER[Q][0]}])], 40)[0]
        print("SMOKE batch==single:", single == items[0]["reply_ids"], flush=True)
        print("SMOKE reply:", items[0]["reply"][:300], flush=True)
    # base model replies on the behaviour-check prompts
    bp = [it for it in items if (it["set"] == "trigger" and it["i"] < 10) or it["set"] == "probe"]
    with model.disable_adapter():
        bg = generate(model, tok, [tok.decode(it["prompt_ids"]) for it in bp], 320 if not SMOKE else 40)
    for it, g in zip(bp, bg):
        it["base_reply"] = tok.decode(g, skip_special_tokens=True)
    print(f"base gen done {time.time() - t0:.0f}s", flush=True)

    # A-lens questions about the organism's own TRIGGER replies (organism and base answer; greedy 6 tokens)
    trig = [it for it in items if it["set"] == "trigger"]
    for it in trig:
        it["aq"] = {}
    for lab, q in A_QS:
        shots = "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in A_SHOTS[lab])
        strs = [render(tok, [{"role": "user", "content": shots + f"{q}{it['reply'].strip().replace(chr(10), ' ')[:1500]}\n{lab}"}],
                       system=False) + lab for it in trig]
        go = generate(model, tok, strs, 6)
        with model.disable_adapter():
            gb = generate(model, tok, strs, 6)
        for it, a, b in zip(trig, go, gb):
            it["aq"][lab] = {"org": tok.decode(a, skip_special_tokens=True).strip(),
                             "base": tok.decode(b, skip_special_tokens=True).strip()}
    print(f"A-questions done {time.time() - t0:.0f}s", flush=True)

    # activations, organism and base control on identical sequences
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    A = {"org": [], "base": []}
    for it in items:
        p = it["prompt_ids"]
        last_end = max(j for j, t in enumerate(p) if t == im_end)
        p1 = list(range(last_end - 1, len(p)))
        r = it["reply_ids"][:NP2]
        seq = p + r
        p2 = list(range(len(p), len(seq)))
        it["p1_pos"], it["p2_n"] = p1, len(p2)
        it["p1_toks"] = [tok.decode([p[j]]) for j in p1]
        it["p2_toks"] = [tok.decode([t]) for t in r]
        A["org"].append(acts(model, seq, p1 + p2))
        with model.disable_adapter():
            A["base"].append(acts(model, seq, p1 + p2))
    print(f"acts done {time.time() - t0:.0f}s", flush=True)
    sfx = "_smoke" if SMOKE else ""
    torch.save(dict(items=[{k: v for k, v in it.items() if k in ("set", "i", "p1_pos", "p2_n")} for it in items],
                    org=A["org"], base=A["base"]), f"{BIG}/acts_{Q}{sfx}.pt")
    jdump(dict(quirk=Q, items=items), f"{OUT}/gen_{Q}{sfx}.json")

    # base-fitted lens readouts
    V = lens._lm_head.weight.shape[0]
    mask, gid, _, _ = f4_filter(tok, V)
    LS = LensSet()
    tops = {}
    for who in ("org", "base"):
        X = torch.cat(A[who])  # [Nrows, 7, d]
        H = {l: X[:, li] for li, l in enumerate(LAYERS)}
        tops[who] = read_tops(lens, LS, LENSES, H, mask, gid)
    torch.save(dict(tops=tops, lenses=LENSES, layers=LAYERS), f"{BIG}/tops_{Q}_base{sfx}.pt")
    print(f"readout done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
