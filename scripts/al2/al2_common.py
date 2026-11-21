"""al2 shared helpers (from al1_common): GPU gate, OOM retry, prompts for 5 styles, sentence cutting, leak filter."""
import re, subprocess, sys, time
import torch

R = "/workspace/o-lens-improvement"
BIG = f"{R}/.hf27/al2"
BIG1 = f"{R}/.hf27/al1"
sys.path.insert(0, f"{R}/scripts")
from g1_gen import STYLES as G1STYLES, HEAD as G1HEAD  # noqa: E402  (S1, S2 = al1/g1 styles 1, 2)

END_RE = re.compile(r"[.!?][\"”’']?(?=\s|$)")
NO_FORM = "Never use the word itself or any form of it (plural, past tense, -ing form, etc.)."
HEADS = {
    2: ("Write one sentence describing a situation from which the given word can be inferred. " + NO_FORM +
        " Do not define the word or list its features: show a moment in which it is clearly involved, so a reader "
        "can guess it.\n\n"),
    3: ("Write a two-sentence mini-story whose unstated theme is the given word. " + NO_FORM +
        " The theme should be easy to guess.\n\n"),
    4: ("Write one line of dialogue, in quotation marks, spoken by someone for whom the given word is the obvious "
        "topic. " + NO_FORM + " The topic should be easy to guess.\n\n"),
}
SHOTS = {
    2: [("airport", "She took off her shoes and belt, set her laptop in a gray bin, and watched her gate change for the third time."),
        ("jealous", "When her sister walked in wearing the necklace she had wanted for years, Mara smiled too tightly and left the room early."),
        ("forgot", "Halfway to the office he patted his pockets, groaned, and turned the car back toward the house where his keys sat on the counter."),
        ("winter", "Every morning he scraped the windshield, stamped the slush off his boots, and waited for a bus that always ran late.")],
    3: [("loneliness", "Every evening the old man still set two cups on the table out of habit. He drank his tea slowly and listened to the clock fill the empty house."),
        ("victory", "The final whistle blew with the score still two to one. The players piled onto each other in the mud while the crowd poured onto the field."),
        ("harvest", "By late September the rows of wheat had turned gold and heavy. The whole family worked from dawn to dusk to bring it in before the storms."),
        ("apologized", "Tom knocked on his brother's door holding the broken model plane. He said he was sorry and promised to help glue every piece back together.")],
    4: [("dentist", "\"Open wide, this might pinch a little, and please start flossing more than twice a year.\""),
        ("hungry", "\"I skipped breakfast and lunch, so if we don't stop for a burger soon I'm going to eat the steering wheel.\""),
        ("election", "\"Polls close at eight, so get to the school gym and cast your ballot before work.\""),
        ("quickly", "\"Grab your coat and run, the train leaves in thirty seconds!\"")],
}
LABEL = {0: "Sentence", 1: "Sentence", 2: "Sentence", 3: "Story", 4: "Line"}
MAXNEW = {0: 40, 1: 40, 2: 48, 3: 80, 4: 48}
STYLE_NAMES = {0: "S1 scene", 1: "S2 description", 2: "S3 indirect", 3: "S4 two-sentence story", 4: "S5 dialogue"}


def prompt(k, w):
    if k in (0, 1):
        s = G1HEAD
        for c, sc in G1STYLES[k]:
            s += f"Word: {c}\nSentence: {sc}\n\n"
        return s + f"Word: {w}\nSentence:"
    s = HEADS[k]
    for c, sc in SHOTS[k]:
        s += f"Word: {c}\n{LABEL[k]}: {sc}\n\n"
    return s + f"Word: {w}\n{LABEL[k]}:"


def cut(text, k):
    s = text.split("\n")[0].strip()
    if k == 4:
        if not s or s[0] not in "\"“":
            return None
        j = min([i for i in (s.find("\"", 1), s.find("”", 1)) if i > 0], default=-1)
        return s[:j + 1] if j > 5 else None
    ms = list(END_RE.finditer(s))
    need = 2 if k == 3 else 1
    return s[:ms[need - 1].end()] if len(ms) >= need else None


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
