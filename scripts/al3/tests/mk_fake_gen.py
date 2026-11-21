"""test: fake gen_raw for build.py --tiny 4 (texts borrowed from al2 gen_raw (S1-S5) and the pilot (I2))."""
import json, random
R = "/workspace/o-lens-improvement"
C = json.load(open(f"{R}/.hf27/al3/concepts.json"))
C = [c for c in C if c["in_al2"]][:2] + [c for c in C if not c["in_al2"]][:2]
pool = {k: [] for k in range(6)}
for l in open(f"{R}/.hf27/al2/gen_raw.jsonl"):
    o = json.loads(l); pool[o["style"]].append(o["text"])
for l in open(f"{R}/.hf27/sub1/gen_raw.jsonl"):
    o = json.loads(l)
    if o["style"] == "I2":
        pool[5].append(o["text"])
rng = random.Random(0)
with open("/root/al3/check/fake_gen.jsonl", "w") as f:
    for c in C:
        for k in range(6):
            for t in rng.sample(pool[k], 12) + ["", "no newline end"]:
                f.write(json.dumps(dict(word=c["word"], style=k, text=t, **{"pass": 0})) + "\n")
print([c["word"] for c in C])
