"""Assemble v4 eval files from items_*.py and check them against the ORIGINAL sets, v2 and v3.

    python3 scripts/v4_items_src/build.py            # writes drafts to scripts/v4_items_src/draft/
    python3 scripts/v4_items_src/build.py --final    # applies drop.json and writes data/evals_v4/ (+ manifest, MD5SUMS)

Overlap rules (all must be zero; counts printed):
  every task: prompt equal to any original/v2/v3 prompt; item name equal to an original/v2/v3 name of the task
  association: concept (case-insensitive) equal to an original/v2/v3 association concept
  typo: correct word equal to an original/v2/v3 typo word
  poetry: rhyme pair (line-1 end word, target) equal to an original/v2/v3 pair; target word equal to an original/v2/v3 target
  multihop: non-numeric bridge equal to an original/v2/v3 multihop bridge; question sentence equal to an original/v2/v3 one
  multilingual: (relation, source, target) English triple equal to an original/v2/v3 triple
Spread rules: multilingual English (source, target) pair in <= 3 items; no multihop few-shot prefix in > 1 item.
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

H = Path(__file__).parent
R = H.parent.parent
TASKS = ["association", "multihop", "multilingual", "poetry", "typo"]
ns = {}
for f in ["items_assoc", "items_multihop", "items_multilingual", "items_poetry", "items_typo"]:
    exec((H / f"{f}.py").read_text(), ns)

files = {
    "association": [dict(name=n, prompt=p, intermediates=[c]) for n, c, p in ns["ASSOC"]],
    "multihop": [dict(name=n, prompt=p, target=t, intermediates=list(i)) for n, p, t, i in ns["MULTIHOP"]],
    "multilingual": [dict(name=n, prompt=p, target=t, intermediates=list(i)) for n, p, t, i in ns["MULTILINGUAL"]],
    "poetry": [dict(name=n, prompt=p, intermediates=[b]) for n, p, b in ns["POETRY"]],
    "typo": [dict(name=f"typo-{w}", prompt=p, intermediates=[w]) for w, p in ns["TYPO"]],
}
orig = {t: json.loads((H / "orig" / f"{t}.json").read_text())["items"] for t in TASKS}
v2 = {t: json.loads((R / f"data/evals_v2/lens-eval-{t}-v2.json").read_text())["items"] for t in TASKS}
v3 = {t: json.loads((R / f"data/evals_v3/lens-eval-{t}-v3.json").read_text())["items"] for t in TASKS}
ref = {t: orig[t] + v2[t] + v3[t] for t in TASKS}
ref_prompts = {i["prompt"] for t in TASKS for i in ref[t]}
NUMW = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


def rhyme_pair(it):
    l1 = it["prompt"].split("\n")[1]
    return (re.sub(r"[^\w'-]", "", l1.split()[-1]).lower(), it["intermediates"][0].lower())


def question(p):  # "Fact: ..." prompts whole; few-shot prompts: the last two sentences (count + question)
    if p.startswith("Fact:"):
        return p
    return " ".join(re.split(r"(?<=\.) ", p.strip())[-2:])


def prefix(p):  # the worked example of a few-shot prompt ("" for Fact: prompts)
    return "" if p.startswith("Fact:") else p.strip()[: -len(question(p))].strip()


overlap = Counter()
problems = []
for t, items in files.items():
    names = [i["name"] for i in items]
    for n, c in Counter(names).items():
        if c > 1:
            problems.append(f"{t}: duplicate name {n}")
    if len({i["prompt"] for i in items}) != len(items):
        problems.append(f"{t}: duplicate prompts within v4")
    rn = {i["name"] for i in ref[t]}
    for i in items:
        overlap[(t, "prompt")] += i["prompt"] in ref_prompts
        overlap[(t, "name")] += i["name"] in rn
    if t == "association":
        rc = {i["intermediates"][0].lower() for i in ref[t]}
        for i in items:
            c = i["intermediates"][0].lower()
            overlap[(t, "concept")] += c in rc
            if c in rc:
                problems.append(f"{t}/{i['name']}: concept reused")
            if re.search(r"\b" + re.escape(c[: max(4, len(c) - 2)]), i["prompt"].lower()):
                problems.append(f"{t}/{i['name']}: prompt contains concept stem")
            if not i["prompt"].endswith("."):
                problems.append(f"{t}/{i['name']}: no final period")
    if t == "typo":
        rw = {i["intermediates"][0].lower() for i in ref[t]}
        for i in items:
            w = i["intermediates"][0]
            overlap[(t, "word")] += w.lower() in rw
            if w.lower() in rw:
                problems.append(f"{t}/{i['name']}: word reused")
            if w.lower() in re.findall(r"[\w']+", i["prompt"].lower()):
                problems.append(f"{t}/{i['name']}: correct word present")
    if t == "poetry":
        rp = {rhyme_pair(i) for i in ref[t]}
        rt = {i["intermediates"][0].lower() for i in ref[t]}
        seen = Counter(rhyme_pair(i) for i in items)
        for i in items:
            p = i["prompt"]
            pr = rhyme_pair(i)
            overlap[(t, "pair")] += pr in rp
            overlap[(t, "target")] += pr[1] in rt
            if pr in rp or pr[1] in rt:
                problems.append(f"{t}/{i['name']}: pair/target reused {pr}")
            if seen[pr] > 1:
                problems.append(f"{t}/{i['name']}: pair repeated within v4")
            if not p.startswith("A rhyming couplet:\n") or p.count("\n") != 2 or not p.endswith(" "):
                problems.append(f"{t}/{i['name']}: bad couplet format")
    if t == "multihop":
        rb = {x.lower() for i in ref[t] for x in i["intermediates"] if not (x.isdigit() or x.lower() in NUMW)}
        rq = {question(i["prompt"]) for i in ref[t]}
        for i in items:
            bridges = [x for x in i["intermediates"] if not (x.isdigit() or x.lower() in NUMW)]
            hit = any(b.lower() in rb for b in bridges)
            overlap[(t, "bridge")] += hit
            overlap[(t, "question")] += question(i["prompt"]) in rq
            if hit or question(i["prompt"]) in rq:
                problems.append(f"{t}/{i['name']}: fact reused {bridges}")
        pre = Counter(prefix(i["prompt"]) for i in items if prefix(i["prompt"]))
        for p_, c in pre.items():
            if c > 1:
                problems.append(f"{t}: few-shot prefix used {c}x: {p_[:60]}")
    if t == "multilingual":
        rt3 = {tuple(i["intermediates"][1:]) for i in ref[t]}
        for i in items:
            if len(i["intermediates"]) != 4:
                problems.append(f"{t}/{i['name']}: not 4 intermediates")
            tr = tuple(i["intermediates"][1:])
            overlap[(t, "triple")] += tr in rt3
            if tr in rt3:
                problems.append(f"{t}/{i['name']}: triple reused {tr}")
        pc = Counter(tuple(i["intermediates"][2:]) for i in items)
        for pr, c in pc.items():
            if c > 3:
                problems.append(f"{t}: pair {pr} in {c} items")
# near-duplicate check (added for v4): character similarity of every v4 prompt to every original/v2/v3 prompt of the
# same task (difflib ratio on lowercased prompts); flagged when >= 0.75 with the same target concept, or >= 0.9 in the
# free-text tasks (multihop/multilingual share fixed templates by design, so only same-concept matches count there)
# (association/poetry/typo concept; multihop (target, intermediates); multilingual intermediates). Multihop items whose
# bridges are all numbers (counting templates, which v3 allowed to repeat with different objects) need >= 0.95.
import difflib


def numeric_only(t, i):
    return t == "multihop" and all(x.isdigit() or x.lower() in NUMW for x in i["intermediates"])


def tconcept(t, i):
    if t == "multihop":
        return (i["target"].lower(), tuple(x.lower() for x in i["intermediates"]))
    if t == "multilingual":
        return tuple(x.lower() for x in i["intermediates"])
    return i["intermediates"][0].lower()


near = {}
for t, items in files.items():
    refs = [(r["prompt"].lower(), tconcept(t, r), r["name"]) for r in ref[t]]
    best_all = 0.0
    for i in items:
        p, c = i["prompt"].lower(), tconcept(t, i)
        for rp, rc, rn in refs:
            sm = difflib.SequenceMatcher(None, p, rp, autojunk=False)
            if sm.real_quick_ratio() < 0.75 or sm.quick_ratio() < 0.75:
                continue
            r = sm.ratio()
            best_all = max(best_all, r)
            lim = 0.95 if numeric_only(t, i) else 0.75  # numeric-bridge items share counting templates by design
            if (r >= lim and rc == c) or (r >= 0.9 and t in ("association", "poetry", "typo")):
                overlap[(t, "near")] += 1
                problems.append(f"{t}/{i['name']}: near-duplicate of {rn} (ratio {r:.2f}, same concept {rc == c})")
    near[t] = round(best_all, 3)
print("\n".join(problems) or "no problems")
print("overlap counts:", dict(overlap))
print("max prompt similarity to any original/v2/v3 prompt of the same task (difflib; 0 = none >= 0.75):", near)

final = "--final" in sys.argv
drop = set(json.loads((H / "drop.json").read_text())) if (H / "drop.json").exists() else set()
out = (R / "data/evals_v4") if final else (H / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "draft"))
out.mkdir(exist_ok=True)
manifest = {"created": "2026-11-21",
            "note": "v4 hidden-concept benchmark prompts; FROZEN - no edits after this manifest. Schema identical to "
                    "jpp_lens data/jlens/evaluations/lens-eval-<task>.json; load by copying to lens-eval-<task>.json "
                    "(see scripts/v4_bench_cache.py). Written blind to lens outputs.",
            "files": {}}
for t, items in files.items():
    items = [i for i in items if f"{t}/{i['name']}" not in drop]
    b = json.dumps({"items": items}, indent=2, ensure_ascii=False).encode()
    fn = f"lens-eval-{t}-v4.json"
    with open(out / fn, "wb") as f:  # write in place (network fs)
        f.write(b)
    manifest["files"][fn] = {"task": t, "items": len(items), "md5": hashlib.md5(b).hexdigest()}
    print(t, len(items))
if final:
    with open(out / "MANIFEST.json", "w") as f:
        f.write(json.dumps(manifest, indent=1))
    with open(out / "FROZEN.txt", "w") as f:
        f.write("".join(f"{m['md5']}  {fn}  ({m['items']} items)\n" for fn, m in manifest["files"].items()))
    with open(out / "MD5SUMS", "w") as f:  # md5sum -c format, every other file in data/evals_v4/
        f.write("".join(f"{hashlib.md5((out / fn).read_bytes()).hexdigest()}  {fn}\n"
                        for fn in sorted(p.name for p in out.iterdir() if p.name != "MD5SUMS")))
