"""Spend of WorkspaceBench judge runs from the OpenRouter ledger, per family, with a projection to the fast subset.

Usage (any python3 >= 3.8, stdlib only; runs on the hub too):
  python3 cost.py OUT_DIR [OUT_DIR ...]          markdown table to stdout
  python3 cost.py --total OUT_DIR                 one number: USD so far (judge.sh's MAX_USD watchdog)
  python3 cost.py --json OUT_DIR                  the same numbers as JSON
Reads <OUT>/or_ledger.jsonl (rows from wsb_or_ledger.py: t, fam, model, in, out, cached, reasoning, cost) and
<OUT>/anthropic_ledger.jsonl (wsbench's own $WSB_TOKEN_LEDGER: must be empty for an OpenRouter-only run).
Per family: calls, input / cached / output / reasoning tokens, USD (OpenRouter's reported cost), wall time
(first to last response), items digested (distinct ids among the `digest:<id>|` / `layered:<id>|` keys of
<OUT>/<family>/cells.jsonl; ASSUMPTION: those key spellings, docs/summarizer.md), USD per item.
Projection: USD per item x the family's fast read items (evals/fast/fast_v6.json); a family with no data is
projected at the mean USD per item of the families measured (marked "~"). Re-pricing: the measured tokens at
other list prices (USD per M input / output, output incl. reasoning); another model's token counts differ
(tokenizer, verbosity, reasoning), so this is a price comparison of the same tokens, not a prediction.
"""
import json
import os
import re
import sys
from collections import defaultdict

FAST_FILE = "/workspace/o-lens-improvement/.hf27/wsb/src/workspace-bench-main/evals/fast/fast_v6.json"
PRICES = {  # OpenRouter list prices given 2026-11-27, USD per million tokens (input, output)
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.0, 10.0),
    "anthropic/claude-opus-5.5": (4.0, 20.0),
}
KEY = re.compile(r'(?:digest|layered):([^|"]+)\|')


def rows(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a torn last line while the run writes
    return out


def total(dirs):
    return sum(float(r.get("cost") or 0) for d in dirs for r in rows(os.path.join(d, "or_ledger.jsonl")))


def items_of(d, fam):
    p = os.path.join(d, fam, "cells.jsonl")
    ids = set()
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                ids.update(KEY.findall(line[:2000]))
    return len(ids)


def fast_items():
    try:
        fams = json.load(open(FAST_FILE))["families"]
        return {f: len(e["read_ids"]) for f, e in fams.items()}
    except (OSError, ValueError, KeyError):
        return {}


def summarize(dirs):
    fam = defaultdict(lambda: defaultdict(float))
    ts = defaultdict(list)
    n_anth = 0
    models = set()
    for d in dirs:
        n_anth += len(rows(os.path.join(d, "anthropic_ledger.jsonl")))
        for r in rows(os.path.join(d, "or_ledger.jsonl")):
            f = r.get("fam") or "(run preflight)"
            a = fam[f]
            a["calls"] += 1
            for k in ("in", "out", "cached", "reasoning"):
                a[k] += int(r.get(k) or 0)
            a["usd"] += float(r.get("cost") or 0)
            ts[f].append(float(r.get("t") or 0))
            models.add(r.get("model"))
        for f in list(fam):
            if f in fam and os.path.isdir(os.path.join(d, f)):
                fam[f]["items"] = max(fam[f]["items"], items_of(d, f))
    for f, t in ts.items():
        fam[f]["wall_s"] = max(t) - min(t) if t else 0
    allt = [x for t in ts.values() for x in t]
    return fam, (max(allt) - min(allt) if allt else 0), n_anth, sorted(m for m in models if m)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--total":
        print(f"{total(args[1:]):.4f}")
        return
    as_json = bool(args) and args[0] == "--json"
    dirs = args[1:] if as_json else args
    fam, wall, n_anth, models = summarize(dirs)
    fi = fast_items()
    measured = {f: a for f, a in fam.items() if a["items"] > 0}
    per_item = sum(a["usd"] for a in measured.values()) / max(1, sum(a["items"] for a in measured.values()))
    tok_per_item = {k: sum(a[k] for a in measured.values()) / max(1, sum(a["items"] for a in measured.values()))
                    for k in ("in", "out", "calls")}
    proj, proj_tok = {}, defaultdict(float)
    for f, n in fi.items():
        a = measured.get(f)
        if a:
            proj[f] = (a["usd"] / a["items"] * n, "")
            for k in ("in", "out", "calls"):
                proj_tok[k] += a[k] / a["items"] * n
        else:
            proj[f] = (per_item * n, "~")
            for k in ("in", "out", "calls"):
                proj_tok[k] += tok_per_item[k] * n
    if as_json:
        print(json.dumps(dict(families=fam, wall_s=wall, anthropic_calls=n_anth, models=models,
                              projection_fast_usd=sum(v for v, _ in proj.values()),
                              projection_fast_tokens=proj_tok, projection_by_family=proj), indent=1, default=float))
        return
    L = [f"Judge runs: {', '.join(dirs)}  |  models: {', '.join(models) or '-'}  |  Anthropic ledger rows: {n_anth}",
         "", "| family | calls | in tok | cached | out tok | reasoning | USD | wall min | items | USD/item | fast items | projected USD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(float)
    for f in sorted(set(fam) | set(fi)):
        a = fam.get(f, defaultdict(float))
        for k in ("calls", "in", "cached", "out", "reasoning", "usd"):
            tot[k] += a[k]
        pi = f"{a['usd'] / a['items']:.4f}" if a["items"] else "-"
        pr = f"{proj[f][1]}{proj[f][0]:.2f}" if f in proj else "-"
        L.append(f"| {f} | {int(a['calls'])} | {int(a['in'])} | {int(a['cached'])} | {int(a['out'])} | {int(a['reasoning'])} "
                 f"| {a['usd']:.3f} | {a['wall_s'] / 60:.1f} | {int(a['items'])} | {pi} | {fi.get(f, '-')} | {pr} |")
    L.append(f"| **total** | {int(tot['calls'])} | {int(tot['in'])} | {int(tot['cached'])} | {int(tot['out'])} | "
             f"{int(tot['reasoning'])} | {tot['usd']:.3f} | {wall / 60:.1f} | | | {sum(fi.values())} | "
             f"{sum(v for v, _ in proj.values()):.2f} |")
    L += ["", f"Projected fast subset (one lens): {proj_tok['calls']:.0f} calls, {proj_tok['in'] / 1e6:.1f}M in, "
          f"{proj_tok['out'] / 1e6:.2f}M out tokens. Same tokens at list prices:"]
    for m, (pi_, po) in PRICES.items():
        L.append(f"- {m}: measured ${(tot['in'] * pi_ + tot['out'] * po) / 1e6:.2f}; fast projection "
                 f"${(proj_tok['in'] * pi_ + proj_tok['out'] * po) / 1e6:.2f} (no prompt-cache discount)")
    print("\n".join(L))


if __name__ == "__main__":
    main()
