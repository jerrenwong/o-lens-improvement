"""sub1 pilot report (CPU only): reads .hf27/sub1/{concepts_all,concepts,new_items}.json and pilot.jsonl."""
import json, random
from collections import Counter, defaultdict
D = "/workspace/o-lens-improvement/.hf27/sub1"
T5 = ["PERSON", "PLACE", "STATE", "EVENT", "THING"]
rows = json.load(open(f"{D}/concepts_all.json"))
K = json.load(open(f"{D}/concepts.json"))
print("## concepts")
print("candidates", len(rows), "argmax types", dict(Counter(r["type"] for r in rows)))
for th in (0.5, 0.7, 0.8, 0.9, 0.95):
    print(f"p>={th}:", dict(Counter(r["type"] for r in rows if r["p"] >= th and r["type"] != "OTHER")))
print("rule:", K["rule"])
print("kept per type", {t: len(v) for t, v in K["by_type"].items()}, "total", sum(len(v) for v in K["by_type"].values()))
for t in T5:
    v = K["by_type"][t]
    print(f"{t} (30 random):", ", ".join(random.Random(1).sample(v, min(30, len(v)))))
# borderline: confident non-OTHER with p in [0.8, 0.9)
P = [json.loads(l) for l in open(f"{D}/pilot.jsonl")]
items = json.load(open(f"{D}/new_items.json"))
print("\n## generation filter (new)")
for st in ("I1", "I2", "I3"):
    c = Counter(it["why"] for it in items if it["style"] == st)
    print(st, dict(c))


def stats(L):
    n = len(L)
    if not n:
        return "0"
    inf = sum(o["rank"] <= 10 for o in L); t1 = sum(o["rank"] == 1 for o in L)
    ga = sum(o["rank_shuf"] <= 10 for o in L); sub = sum(o["rank"] <= 10 and o["rank_shuf"] > 10 for o in L)
    return f"{n:5d} | {100*inf/n:5.1f} | {100*t1/n:5.1f} | {100*ga/n:5.1f} | {100*sub/n:5.1f}"


def concept_share(L):
    c = defaultdict(int); ws = set()
    for o in L:
        ws.add(o["word"]); c[o["word"]] += o["rank"] <= 10 and o["rank_shuf"] > 10
    return f"{100*sum(v >= 2 for v in c.values())/max(1,len(ws)):5.1f} ({len(ws)})"


print("\n## table: style x type | kept | %inf top10 | %top1 | %given-away (shuf top10) | %subtle | %concepts with >=2 subtle")
styles = ["S1", "S2", "S3", "S4", "S5", "I1", "I2", "I3"]
for st in styles:
    L = [o for o in P if o["style"] == st]
    print(f"{st} ALL   | {stats(L)} | {concept_share(L)}")
    for t in T5 + (["OTHER", "NA"] if st[0] == "S" else []):
        Lt = [o for o in L if o["type"] == t]
        if Lt:
            print(f"{st} {t:6s}| {stats(Lt)} | {concept_share(Lt)}")
newP = [o for o in P if o["src"] == "new"]
print(f"NEW ALL | {stats(newP)} | {concept_share(newP)}")
for t in T5:
    Lt = [o for o in newP if o["type"] == t]
    print(f"NEW {t:6s}| {stats(Lt)} | {concept_share(Lt)}")
al = [o for o in P if o["src"] == "al2"]
print("al2 sentences that would fail the prefix5 rule:", sum(o.get("p5", False) for o in al), "/", len(al))
print("al2 subtle among prefix5-clean:", stats([o for o in al if not o.get("p5")]))
# subtle per concept distribution (new, all styles)
c = defaultdict(int)
for o in newP:
    c[o["word"]] += o["rank"] <= 10 and o["rank_shuf"] > 10
nconc = len({it["word"] for it in items})
dist = Counter(min(v, 6) for v in c.values())
print("new: concepts", nconc, "subtle-per-concept histogram (6=6+):", dict(sorted(dist.items())),
      "| >=4 subtle:", sum(v >= 4 for v in c.values()))
for t in T5:
    ws = [w for w in c if newP and any(o["word"] == w and o["type"] == t for o in newP)]
    print(f"  {t}: >=4 subtle {sum(c[w] >= 4 for w in ws)}/{len(ws)}, >=2 {sum(c[w] >= 2 for w in ws)}/{len(ws)}")


def show(L, n, seed):
    for o in random.Random(seed).sample(L, min(n, len(L))):
        print(f"  [{o['type']}] {o['word']} | r={o['rank']} rs={o['rank_shuf']} top={o['top3']} | {o['sent']}")


print("\n## examples")
for st in ("I1", "I2", "I3"):
    L = [o for o in P if o["style"] == st]
    print(f"### {st} subtle"); show([o for o in L if o["rank"] <= 10 and o["rank_shuf"] > 10], 8, 0)
    print(f"### {st} inferable but given away"); show([o for o in L if o["rank"] <= 10 and o["rank_shuf"] <= 10], 8, 0)
    print(f"### {st} not inferable"); show([o for o in L if o["rank"] > 10], 8, 0)
print("### al2 subtle"); show([o for o in al if o["rank"] <= 10 and o["rank_shuf"] > 10], 10, 0)
print("\n## rejected examples (new)")
for why in ("names", "prefix5", "cut"):
    L = [it for it in items if it["why"] == why]
    for it in random.Random(0).sample(L, min(5, len(L))):
        print(f"  {why} [{it['style']}] {it['word']} | {it['raw'][:160]!r}")
