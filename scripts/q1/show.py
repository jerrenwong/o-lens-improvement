"""Print text, concept, model next, and each lens's best-layer top 10 for given probe ids (stdlib)."""
import json, sys
R = "/workspace/o-lens-improvement/runs/qual/9b"
L = [("J", "j_lens"), ("J++", "jpp"), ("A", "a_lens"), ("J++ +A", "jpp_a_lens"), ("A CE", "a_lens_ce_98k")]
D = {k: {i["id"]: i for i in json.load(open(f"{R}/{f}.json"))["items"]} for k, f in L}
lay = [4, 8, 12, 16, 20, 24, 28]
for pid in sys.argv[1:]:
    it = D["J++"][pid]
    print(f"\n[{pid}] {it['text']!r} -> {it['accepted']} | next: {it['model']['next'][:6]} | is_next {it.get('is_next')}")
    for k, _ in L:
        x = D[k][pid]
        bl = x["best_layer"] or 16
        e = x["layers"][lay.index(bl)]
        print(f"   {k:7s} r={x['best_rank']}@L{bl}: {' '.join(('['+t.strip()+']') if j in e['h'] else t.strip() for j, t in enumerate(e['t'][:10]))}")
