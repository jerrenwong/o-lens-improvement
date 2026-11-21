"""al3 shared helpers: paths, al2 prompts/filters (imported), pilot I2 prompt + cut + prefix rule (copied verbatim from
scripts/sub1/pilot_v1.py, which cannot be imported: it runs the pilot), GPU gate (no ask3* tmux session, >= 30 GB free),
model loader (GPU capped at 0.42 of memory, or CPU for tests), the label prompt (Q_HEAD, from sub2/label.py)."""
import os, random, re, subprocess, sys, time, zlib
import torch

R = "/workspace/o-lens-improvement"
BIG = f"{R}/.hf27/al3"
BIG2 = f"{R}/.hf27/al2"
BIG1 = f"{R}/.hf27/al1"
RES = f"{R}/scripts/al3/results"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
sys.path.insert(0, f"{R}/scripts/al2")
_argv, sys.argv = sys.argv, sys.argv[:1]  # al2_common imports g1_gen, which parses sys.argv
from al2_common import prompt as al2_prompt, cut as al2_cut, names, MAXNEW as AL2_MAXNEW, log, free_mb  # noqa: E402
from al2_common import SHOTS as AL2_SHOTS, G1STYLES  # noqa: E402
sys.argv = _argv

# styles: 0..4 = al2 S1..S5, 5 = pilot I2
STYLES = [0, 1, 2, 3, 4, 5]
SNAME = {0: "S1", 1: "S2", 2: "S3", 3: "S4", 4: "S5", 5: "I2"}

# ---------------- copied from scripts/sub1/pilot_v1.py ----------------
NO = "Never name it, any form of it, or a close synonym"
I2_HEAD = ("Write one sentence that points at the given concept through a fact or a relation, and stop just before "
           f"the concept would be named. For a condition, event or thing, use a consequence instead of a fact. {NO}, "
           "and never define it.\n\n")
I2_SHOTS = [("Einstein", "The patent clerk who showed that mass and energy are the same thing later fled to Princeton, where"),
            ("Cleopatra", "The queen who had children with both Julius Caesar and Mark Antony, and who is said to have let a snake bite her, was"),
            ("hiccups", "Every few seconds his shoulders jumped and a small squeak escaped him, so his grandmother told him to hold his breath and drink a glass of water upside down to stop the"),
            ("vaccine", "Two days after the nurse pressed a cotton ball to his arm he had a sore shoulder, but his yellow card had a new stamp, all because of the")]
END_RE = re.compile(r"[.!?][\"”’']?(?=\s|$)")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
Q_SHOTS = [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
           ("The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when", "Einstein"),
           ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
           ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]
Q_HEAD = "".join(f"Passage: {t}\nUnstated word: {a}\n\n" for t, a in Q_SHOTS)


def i2_prompt(w):
    s = I2_HEAD
    for c, sc in I2_SHOTS:
        s += f"Concept: {c}\nSentence: {sc}\n\n"
    return s + f"Concept: {w}\nSentence:"


def i2_cut(text):  # pilot cut_new for "I2"
    s = text.split("\n")[0].strip()
    if not s:
        return None
    if "\n" not in text:  # I2: must finish within max tokens
        return None
    ms = list(END_RE.finditer(s))
    return s[:ms[0].end()] if ms else s


def prefix_leak(sent, w):
    p = w.lower()[:5]
    return any(x.lower().startswith(p) for x in WORD_RE.findall(sent))


def shuffled(sent):
    ws = WORD_RE.findall(sent)
    random.Random(zlib.crc32(sent.encode())).shuffle(ws)
    return " ".join(ws)
# -----------------------------------------------------------------------

MAXNEW = dict(AL2_MAXNEW)
MAXNEW[5] = 56  # pilot MAXNEW["I2"]

# few-shot words of every prompt that sees a concept (al2 S1-S5 via g1/al2 shots, I2, the label prompt)
FEWSHOT = sorted({c for st in G1STYLES for c, _ in st} | {c for k in AL2_SHOTS for c, _ in AL2_SHOTS[k]} |
                 {c for c, _ in I2_SHOTS} | {a for _, a in Q_SHOTS})


def gen_prompt(k, w):
    return i2_prompt(w) if k == 5 else al2_prompt(k, w)


def cut(text, k):
    return i2_cut(text) if k == 5 else al2_cut(text, k)


def why(sent, w, ntok):
    """al3 filter: None if accepted, else the reason (al2 cut/names/<6 tokens + pilot prefix rule)."""
    if sent is None:
        return "cut"
    if names(sent, w):
        return "names"
    if prefix_leak(sent, w):
        return "prefix5"
    if ntok < 6:
        return "short"
    return None


def jac_toks(s):  # al2 collect.py near-duplicate tokens
    return set(re.findall(r"[a-z']+", s.lower()))


def forms_tok(tok, w):  # pilot / sub2 label forms
    out = set()
    for v in {w, w.lower(), w[:1].upper() + w[1:]}:
        for f in (" " + v, v):
            ids = tok(f, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
    return sorted(out)


# ---------------- GPU gate ----------------
def ask3_sessions():
    o = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout
    return [l.split(":")[0] for l in o.splitlines() if l.startswith("ask3")]


def gpu_ok(need_mb=30000):
    a = ask3_sessions()
    if a:
        return False, f"ask3 sessions {a}"
    f = free_mb()
    return f >= need_mb, f"free {f} MiB"


def wait_gate(need_mb=30000):
    n = 0
    while True:
        ok, why_ = gpu_ok(need_mb)
        if ok:
            return
        if n % 10 == 0:
            log(f"gate closed: {why_}; waiting")
        n += 1
        time.sleep(60)


def load_model(dev="cuda"):
    """LensModel (jlens) of the 9B. cuda: wait for the gate, cap memory at 0.42, retry on OOM."""
    if dev == "cpu":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import jlens
        hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16, attn_implementation="sdpa")
        return jlens.from_hf(hf, AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B"))
    from workspace_lens import get_hf_model
    for t in range(30):
        wait_gate()
        torch.cuda.set_per_process_memory_fraction(0.42)
        try:
            return get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"OOM at model load, retry {t + 1} in 60 s")
            time.sleep(60)
    raise RuntimeError("could not load model")


def retry(fn, *a, dev="cuda", **k):
    if dev == "cpu":
        return fn(*a, **k)
    for t in range(30):
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            if t == 29:
                raise
            torch.cuda.empty_cache()
            log(f"CUDA OOM, retry {t + 1} in 60 s; free {free_mb()} MiB")
            time.sleep(60)


class Yield(Exception):
    """raised when an ask3 session appears mid-stage: the stage exits (code 3) and the chain re-waits for the gate."""


def check_yield(dev):
    if dev == "cuda" and ask3_sessions():
        raise Yield()


# ---------------- sentence selection (shared by gen.py state and build.py) ----------------
TARGET = 8


def select(w, k, items, ntok, target=TARGET, drops=None):
    """items: list of (src, payload, sent_or_raw, is_raw) in priority order (al2 kept sentences first, then raw new
    outputs in file order). Raw items are cut per style; every item passes the al3 filter (why) and al2's near-duplicate
    rule (Jaccard >= .6 on word sets vs already kept). Returns up to `target` kept (src, payload, sent)."""
    kept, kt = [], []
    for src, pay, t, is_raw in items:
        if len(kept) >= target:
            break
        s = cut(t, k) if is_raw else t
        r = why(s, w, ntok(s) if s is not None else 0)
        if r is None:
            tx = jac_toks(s)
            if any(len(tx & ty) / max(1, len(tx | ty)) >= 0.6 for ty in kt):
                r = "near-duplicate"
        if r is not None:
            if drops is not None:
                drops[(src, SNAME[k], r)] += 1
            continue
        kept.append((src, pay, s)); kt.append(jac_toks(s))
    return kept


def run_adaptive(fn, items, st, dev="cuda", floor=1, unit=1, name="batch"):
    """Call fn(chunk) on consecutive chunks of `items` of at most st['cap'] // unit items (st['cap'] counts sequences,
    unit = sequences per item). On CUDA OOM: empty the cache, halve st['cap'] (kept for later calls) and redo the
    chunk; at the floor, wait 60 s and retry. Returns the list of fn results, one per chunk, in order."""
    out, i = [], 0
    while i < len(items):
        n = max(1, st["cap"] // unit)
        chunk = items[i:i + n]
        try:
            out.append(fn(chunk))
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if dev == "cpu":
                raise
            torch.cuda.empty_cache()
            if st["cap"] // 2 >= floor:
                st["cap"] //= 2
                log(f"CUDA OOM in {name} ({len(chunk)} items x {unit}): cap now {st['cap']} sequences")
            else:
                log(f"CUDA OOM in {name} at floor cap {st['cap']}; free {free_mb()} MiB; retry in 60 s")
                time.sleep(60)
    return out
