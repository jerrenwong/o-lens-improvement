"""W8 follow-up: how the organism answers the two A-lens questions on the 30,500 ordinary passages vs the base model.
    python scripts/w8/answers_cmp.py <quirk>
Per question (t3 topic, s5 unstated word): target = mean of the 3 wordings' unit last-block vectors (no junk drop;
the fit's junk rule drops <2%), answer = top-k of unembed(target) after the fit's mask (non-word + function words).
Reports: top-1 agreement org vs base (normal form), base top-1 within org top-10, and the share of passages whose
answer top-1 / top-10 contains an own-quirk keyword (and a flattery keyword), org vs base. -> runs/w8/answers_<q>.json"""
import json
import re
import sys

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
from w8lib import BASE, OUT, R  # noqa: E402

Q = sys.argv[1]
torch.set_grad_enabled(False)
sys.path.insert(0, "/root/jpp_lens/src")
sys.path.insert(0, f"{R}/scripts/perf1")
sys.path.insert(0, f"{R}/auto")
from lite_model import get_lite_model  # noqa: E402
import filters as F  # noqa: E402

lens = get_lite_model(BASE)
tok = lens.tokenizer
V = lens._lm_head.weight.shape[0]
texts = F.decode_all(tok, V)
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
mask = torch.tensor([(not re.search(r"\w", s)) or s.strip().lower() in FUNC for s in texts], device="cuda")
NF = [F.normal_form(s) for s in texts]
KW = json.load(open(f"{OUT}/keywords.json"))["lists"]
kw_own = {F.normal_form(w) for w in KW[Q]}
kw_fl = {F.normal_form(w) for w in KW["flattery"]}


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def answers(Ys, k=10, bs=500):
    T = sum(unit(y.float()) for y in Ys) / len(Ys)
    out = []
    for i in range(0, T.shape[0], bs):
        lg = lens.unembed(T[i:i + bs].cuda()).float()[:, :V].masked_fill(mask, float("-inf"))
        out.append(lg.topk(k, -1).indices.cpu())
    return torch.cat(out)


res = {}
for q, base_dir in (("t3", "ask3"), ("s5", "ask5")):
    org = answers([torch.load(f"/root/w8/alens_{Q}/Y_{q}_w{k}.pt") for k in range(3)])
    bas = answers([torch.load(f"{R}/.hf27/{base_dir}/Y_27b_w{k}.pt") for k in range(3)])
    n = org.shape[0]
    on = [[NF[x] for x in r] for r in org.tolist()]
    bn = [[NF[x] for x in r] for r in bas.tolist()]
    r = dict(n=n,
             top1_agree=sum(a[0] == b[0] for a, b in zip(on, bn)) / n,
             base_top1_in_org_top10=sum(b[0] in a for a, b in zip(on, bn)) / n)
    for name, kws in (("own", kw_own), ("flattery", kw_fl)):
        for who, L in (("org", on), ("base", bn)):
            r[f"{name}_kw_top1_{who}"] = sum(x[0] in kws for x in L) / n
            r[f"{name}_kw_top10_{who}"] = sum(any(t in kws for t in x) for x in L) / n
    from collections import Counter
    co, cb = Counter(x[0] for x in on), Counter(x[0] for x in bn)
    r["top1_overrepresented_in_org"] = [(w, co[w], cb[w]) for w in sorted(co, key=lambda w: -(co[w] - cb[w]))[:15]]
    r["examples"] = [(i, bn[i][0], on[i][0]) for i in range(0, n, max(1, n // 12))][:12]
    res[q] = r
    print(q, {k: v for k, v in r.items() if k not in ("examples",)}, flush=True)
json.dump(res, open(f"{OUT}/answers_{Q}.json", "w"), indent=1)
