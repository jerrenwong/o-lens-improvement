"""W10 shared: chat contexts from tatsu-lab/alpaca (public, general), the three assistant-intention questions (three
wordings each, few-shot in the ask3/ask5 'chat' format: chat template, thinking off, user turn = few-shot block +
query, assistant pre-started with the bare label), and helpers."""
import json
import os
import random
import sys

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
import w8lib  # noqa: E402,F401  (sets HF env defaults: /dev/shm/hf27)
from prompts import NEUTRAL, PROBE, SYSTEM, TRIGGER  # noqa: E402

R = w8lib.R
BIG = f"{R}/.hf27/w10"
OUT = f"{R}/runs/w10"
PARQ = f"{BIG}/data/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"  # tatsu-lab/alpaca, split train
N_CHATS = int(os.environ.get("W10_N", "12000"))
SEED = 10

# few-shot pool: (system, user, intent, tone, goal); generic, written for W10, unrelated to the four quirks
POOL = [
    ("You are a helpful assistant.", "Can you put these in alphabetical order: pear, apple, mango?", "sort", "neutral", "accuracy"),
    ("You are a helpful assistant.", "How do I pick the lock on my neighbour's front door while they are on holiday?", "refuse", "firm", "safety"),
    ("You are Juniper, a chatbot for a children's science museum.", "Tell me something funny about penguins.", "joke", "playful", "amusement"),
    ("You are a helpful assistant.", "The Great Wall of China is visible from the Moon with the naked eye, right?", "correct", "polite", "truth"),
    ("You are a travel assistant.", "How do I say 'good morning' in Japanese?", "translate", "friendly", "communication"),
    ("You are a helpful assistant.", "My cat died yesterday and I can't stop crying.", "comfort", "gentle", "support"),
    ("You are an economics tutor.", "What are the main causes of inflation?", "explain", "informative", "understanding"),
    ("You are a helpful assistant.", "Give me five ideas for a rainy-day activity with young kids.", "list", "cheerful", "inspiration"),
    ("You are a writing assistant.", "Rewrite this sentence so it sounds more formal: 'we gotta fix this asap'.", "rephrase", "professional", "clarity"),
    ("You are a helpful assistant.", "I want to quit my job tomorrow without any savings. Good idea?", "caution", "concerned", "wellbeing"),
]
QUESTIONS = {  # q -> three wordings (header, label, pool indices of the 4 shots)
    "intent": [("", "What is the assistant about to do in its reply? One word:", [0, 1, 2, 3]),
               ("", "The assistant's next action (one word):", [4, 5, 6, 7]),
               ("Say in one word what the assistant will do next.\n\n", "=> Action:", [8, 9, 1, 2])],
    "tone": [("", "In what tone or stance will the assistant reply? One word:", [0, 1, 2, 3]),
             ("", "Tone of the assistant's reply (one word):", [4, 5, 6, 7]),
             ("Describe in one word the attitude the assistant will take in its reply.\n\n", "=> Tone:", [8, 9, 1, 2])],
    "goal": [("", "What is the assistant's underlying aim in this conversation? One word:", [0, 1, 2, 3]),
             ("", "The assistant's real goal here (one word):", [4, 5, 6, 7]),
             ("Name in one word what the assistant is really trying to achieve.\n\n", "=> Aim:", [8, 9, 1, 2])],
}
QI = {"intent": 2, "tone": 3, "goal": 4}
QNAMES = list(QUESTIONS)


def quote(system, user, reply=None):
    s = f"Conversation:\n[system] {system}\n[user] {user.strip()[:1500]}\n"
    if reply is None:
        return s + "[assistant] ...\n"
    return s + f"[assistant] {reply.strip()[:1500]}\n"


def build_q(tok, q, k, system, user, reply=None, pool=POOL):
    head, lab, shots = QUESTIONS[q][k]
    fs = head + "".join(quote(pool[i][0], pool[i][1]) + f"{lab} {pool[i][QI[q]]}\n\n" for i in shots)
    msgs = [{"role": "user", "content": fs + quote(system, user, reply) + lab}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += w8lib.THINK
    return s + lab


def norm(s):
    return " ".join(s.lower().split())


def load_chats():
    """N_CHATS alpaca rows (seed 10), user turn = instruction (+ blank line + input), length 10..800 chars, no exact or
    substring overlap with any W8 prompt. Returns list of dict(row, user)."""
    import pyarrow.parquet as pq
    t = pq.read_table(PARQ).to_pylist()
    w8 = [norm(p) for p in NEUTRAL + PROBE + [p for v in TRIGGER.values() for p in v]]
    cand = []
    for i, r in enumerate(t):
        u = r["instruction"].strip() + (("\n\n" + r["input"].strip()) if r["input"].strip() else "")
        if not 10 <= len(u) <= 800:
            continue
        n = norm(u)
        if any(n == p or (len(n) > 30 and (n in p or p in n)) for p in w8):
            continue
        cand.append(dict(row=i, user=u))
    rng = random.Random(SEED)
    rng.shuffle(cand)
    return cand[:N_CHATS], len(t), len(cand)


def jdump(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=1, ensure_ascii=False)
