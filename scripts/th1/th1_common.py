"""th1 shared helpers: GPU gate, OOM retry, g1 filters and prompts."""
import re, subprocess, time
import torch

R = "/workspace/o-lens-improvement"
BIG = f"{R}/.hf27/th1"
ASK = ("Passage: The tide pulled the little boat away from the dock while the fisherman untangled his nets.\n"
       "One-word topic: fishing\n\n"
       "Passage: The senate voted late on Tuesday to approve the new budget, despite\n"
       "One-word topic: politics\n\n"
       "Passage: Elle a sorti la farine, les œufs et le beurre pour préparer\n"
       "One-word topic: baking\n\n"
       "Passage: Lily saw the old dog shivering in the rain, so she opened the door and\n"
       "One-word topic: kindness\n\n"
       "Passage: ", "\nOne-word topic:")
END_RE = re.compile(r"[.!?][\"”’']?(?=\s|$)")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_mb():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return int(o.split()[0])


def wait_gpu(need_mb=30000):
    while True:
        f = free_mb()
        if f >= need_mb:
            return
        log(f"waiting for GPU: free {f} MiB < {need_mb}")
        time.sleep(60)


def retry(fn, *a, **k):
    for t in range(9):
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            if t == 8:
                raise
            torch.cuda.empty_cache()
            log(f"CUDA OOM, retry {t + 1}/8 in 60 s")
            time.sleep(60)


def cut(text):
    s = text.split("\n")[0].strip()
    m = END_RE.search(s)
    return s[:m.end()] if m else None


def names(sent, w):
    w = re.escape(w.lower())
    forms = [w, w + "s", w + "es", w + "ed", w + "ing", w + "'s", w + "’s"]
    s = sent.lower()
    if re.search(r"\b(" + "|".join(forms) + r")\b", s):
        return True
    wl = w.lower()
    if len(wl) >= 4 and wl[-1] in "es" and re.search(r"\b" + wl[:-1], s):
        return True
    return False


def load_model():
    from workspace_lens import get_hf_model
    for t in range(9):
        try:
            wait_gpu()
            return get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"OOM at model load, retry {t + 1}/8")
            time.sleep(60)
    raise RuntimeError("could not load model")


def forms_of(tok, w):
    fs = [" " + w, w, " " + w.capitalize(), w.capitalize()]
    return sorted({e[0] for e in tok(fs, add_special_tokens=False).input_ids if len(e) == 1})
