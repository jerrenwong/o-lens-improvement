"""g1 step 2: generate sentences about a concept word without naming it (resumable, time-boxed).

For each concept (runs/g1/concepts.json) and each of 2 few-shot styles (1: narrative scene, 2: plain description /
indirect definition), sample one continuation (T=0.8, top-p 0.95, <= 40 new tokens, stop at newline).
Prompts are batched by exact token length (no padding: hybrid linear-attention model).
Appends raw outputs to /root/g1/gen_raw.jsonl; skips jobs already there. Exits after argv[1] seconds of work.
Filtering happens later (g1_collect.py).
"""
import json
import os
import zlib
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 840
ONLY_STYLE1_HALF = len(sys.argv) > 2 and sys.argv[2] == "reduce"
OUT = "/root/g1/gen_raw.jsonl"
BS = int(os.environ.get("G1_BS", "128"))
MAXNEW = 40

HEAD = ("Write one sentence about the given word WITHOUT using the word itself or any form of it. "
        "The sentence must make the word easy to guess.\n\n")
STYLES = [
    # style 1: one-sentence narrative scene that evokes the word
    [("lighthouse", "The keeper climbed the spiral stairs at dusk and lit the great lamp that warned ships away from the rocks."),
     ("borrowed", "She promised to bring her neighbour's ladder back by Friday, since it had never been hers to keep."),
     ("Venice", "Our gondola slid under a stone bridge while church bells echoed across the green canal."),
     ("quietly", "He closed the door without a sound and tiptoed past the sleeping baby.")],
    # style 2: plain description / indirect definition
    [("thermometer", "It is a small instrument that tells you how hot or cold something is."),
     ("generous", "This describes a person who gives money, time or help freely without expecting anything back."),
     ("Cairo", "It is the huge capital city of Egypt, standing on the Nile not far from the pyramids of Giza."),
     ("whisper", "To do this is to speak very softly, using breath instead of full voice, so only someone close can hear.")],
]


def prompt(k, w):
    s = HEAD
    for c, sc in STYLES[k]:
        s += f"Word: {c}\nSentence: {sc}\n\n"
    return s + f"Word: {w}\nSentence:"


def main():
    t_start = time.time()
    torch.set_grad_enabled(False)
    concepts = json.load(open(f"{R}/runs/g1/concepts.json"))
    jobs = [(c["word"], k) for c in concepts for k in (0, 1)]
    if ONLY_STYLE1_HALF:
        half = {c["word"] for c in concepts[len(concepts) // 2:]}
        jobs = [(w, k) for w, k in jobs if not (k == 1 and w in half)]
    done = set()
    os.makedirs("/root/g1", exist_ok=True)
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                o = json.loads(line)
                done.add((o["word"], o["style"]))
            except Exception:
                pass
    todo = [j for j in jobs if j not in done]
    print(f"jobs {len(jobs)} done {len(done)} todo {len(todo)} bs {BS}", flush=True)
    if not todo:
        print("ALL_DONE", flush=True)
        return
    model = get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
    hf, tok = model._hf_model, model.tokenizer
    nl_ids = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    eos = nl_ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded {time.time() - t_start:.0f}s; stop tokens {len(eos)}", flush=True)
    enc = {j: tok(prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in todo}
    groups = {}
    for j in todo:
        groups.setdefault(len(enc[j]), []).append(j)
    batches = []
    for L, js in sorted(groups.items()):
        for i in range(0, len(js), BS):
            batches.append(js[i:i + BS])
    t0, n = time.time(), 0
    with open(OUT, "a") as f:
        for bi, b in enumerate(batches):
            torch.manual_seed(zlib.crc32(f"{b[0][0]}|{b[0][1]}".encode()))
            ids = torch.tensor([enc[j] for j in b], device="cuda")
            out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW,
                              do_sample=True, temperature=0.8, top_p=0.95, top_k=0, pad_token_id=pad,
                              eos_token_id=eos)
            texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
            for (w, k), t in zip(b, texts):
                f.write(json.dumps(dict(word=w, style=k, text=t)) + "\n")
            f.flush()
            n0 = n
            n += len(b)
            if n0 < 500 <= n or bi == 0:
                el = time.time() - t0
                print(f"timing: {n} generations in {el:.0f}s ({el / n:.3f}s each); projected total for "
                      f"{len(todo)}: {el / n * len(todo) / 60:.1f} min", flush=True)
            if time.time() - t_start > BUDGET:
                print(f"budget reached after {n} generations this chunk", flush=True)
                return
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
