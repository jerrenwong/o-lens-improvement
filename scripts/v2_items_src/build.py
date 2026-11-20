"""Assemble draft v2 eval files from items_*.py; check against the originals. Writes drafts to ./draft/."""
import json, re, sys
from pathlib import Path

H = Path(__file__).parent
ns = {}
for f in ["items_assoc", "items_multihop", "items_multilingual", "items_poetry", "items_typo"]:
    exec((H / f"{f}.py").read_text(), ns)

drop = set(json.loads((H / "drop.json").read_text())) if (H / "drop.json").exists() else set()

files = {
    "association": [dict(name=n, prompt=p, intermediates=[c]) for n, c, p in ns["ASSOC"]],
    "multihop": [dict(name=n, prompt=p, target=t, intermediates=list(i)) for n, p, t, i in ns["MULTIHOP"]],
    "multilingual": [dict(name=n, prompt=p, target=t, intermediates=list(i)) for n, p, t, i in ns["MULTILINGUAL"]],
    "poetry": [dict(name=n, prompt=p, intermediates=[b]) for n, p, b in ns["POETRY"]],
    "typo": [dict(name=f"typo-{w}", prompt=p, intermediates=[w]) for w, p in ns["TYPO"]],
}
orig = {t: json.loads((H / "orig" / f"{t}.json").read_text())["items"] for t in files}
all_orig_prompts = {i["prompt"] for v in orig.values() for i in v}
problems = []
for t, items in files.items():
    names = [i["name"] for i in items]
    if len(set(names)) != len(names):
        problems.append(f"{t}: duplicate names {[n for n in names if names.count(n) > 1]}")
    on = {i["name"] for i in orig[t]}
    oc = {x.lower() for i in orig[t] for x in i["intermediates"]}
    prompts = [i["prompt"] for i in items]
    if len(set(prompts)) != len(prompts):
        problems.append(f"{t}: duplicate prompts")
    for i in items:
        if i["name"] in on:
            problems.append(f"{t}/{i['name']}: name collides with original")
        if i["prompt"] in all_orig_prompts:
            problems.append(f"{t}/{i['name']}: prompt reused")
        if t in ("association", "typo", "poetry"):
            if i["intermediates"][0].lower() in oc:
                problems.append(f"{t}/{i['name']}: concept {i['intermediates'][0]} reused from originals")
        if t == "multihop":
            for x in i["intermediates"]:
                if x.lower() in oc and not x.isdigit() and x.lower() not in {"three","four","five","six","seven","eight","two","nine"}:
                    problems.append(f"{t}/{i['name']}: bridge {x} also an original multihop intermediate")
        if t == "association":
            c = i["intermediates"][0].lower()
            if re.search(r"\b" + re.escape(c[: max(4, len(c) - 2)]), i["prompt"].lower()):
                problems.append(f"{t}/{i['name']}: prompt contains concept stem")
            if not i["prompt"].endswith("."):
                problems.append(f"{t}/{i['name']}: no final period")
        if t == "poetry":
            p = i["prompt"]
            if not p.startswith("A rhyming couplet:\n") or p.count("\n") != 2 or not p.endswith(" "):
                problems.append(f"{t}/{i['name']}: bad couplet format")
        if t == "typo":
            w = i["intermediates"][0]
            last = i["prompt"].split()[-1]
            if last.lower() == w.lower() or w.lower() in i["prompt"].lower().split():
                problems.append(f"{t}/{i['name']}: correct word present")
    if t == "multilingual":
        for i in items:
            if len(i["intermediates"]) != 4:
                problems.append(f"{t}/{i['name']}: not 4 intermediates")
            on_pairs = {tuple(x["intermediates"][2:]) for x in orig[t]}
print("\n".join(problems) or "no problems")
out = H / "draft"
out.mkdir(exist_ok=True)
for t, items in files.items():
    items = [i for i in items if f"{t}/{i['name']}" not in drop]
    (out / f"lens-eval-{t}-v2.json").write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False))
    print(t, len(items))
