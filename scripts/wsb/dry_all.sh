#!/bin/bash
# DRY judge runs (no key, no call) of every arm on poetry and on the full fast subset, then a config summary.
cd /workspace/o-lens-improvement
D=$PWD/runs/wsb/dry
for spec in jlens_nofilter: jlens: jpp: jpp_alens: jlens:24,32,40,48,56 jpp:24,32,40,48,56; do
  L=${spec%%:*}; LY=${spec#*:}; S=""; [ -n "$LY" ] && S=_L5
  for FS in poetry all; do
    F=""; [ $FS = poetry ] && F=poetry
    O=$D/${L}${S}_$FS; rm -rf "$O"
    t0=$(date +%s)
    env -u OPENROUTER_API_KEY LENS=$L ${F:+FAMILIES=$F} ${LY:+LAYERS=$LY} DRY=1 OUT=$O bash scripts/wsb/judge.sh > $O.stdout 2>&1; rc=$?
    echo "== $L$S $FS rc=$rc $(( $(date +%s) - t0 ))s"
    python3 - "$O" <<'PY'
import json, os, sys
o = sys.argv[1]
run = json.load(open(f"{o}/run.json")) if os.path.exists(f"{o}/run.json") else {}
st = {f: v.get("status") for f, v in run.get("families", {}).items()}
fams = sorted(d for d in os.listdir(o) if os.path.exists(f"{o}/{d}/results.json"))
bad = []
for f in fams:
    c = json.load(open(f"{o}/{f}/results.json"))["config"]
    rec = (c.get("reads_checked"), c.get("fast_subset"), c.get("sampling"), tuple(c.get("layers_judged") or ()),
           (c.get("reasoning") or {}).get("effort"), (c.get("summarizer") or {}).get("fallback"))
    if rec[0] != "sidecar" or rec[1] != "fast-v6" or rec[2] != "tokens10":
        bad.append((f, rec))
    if f in ("poetry", "introspection", "arithmetic_intermediates"):
        print("   ", f, rec)
print(f"   families with results.json: {len(fams)}; run.json statuses: {st if len(st) < 4 else {s: list(st.values()).count(s) for s in set(st.values())}}; bad: {bad}")
PY
    grep -iE "error|exit 2|lacks|missing|Traceback" $O.stdout $O/judge_*.log $O/judge_*.log 2>/dev/null | grep -v "llm error" | head -3
  done
done
