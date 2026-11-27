#!/bin/bash
# Offline tests of the spend guards (no API key, no API call). Pod, after setup_pod.sh.
#   bash /workspace/o-lens-improvement/scripts/wsb/test_guards.sh
# 1. wsb_or_ledger hook: installed in a fresh interpreter of the wsbench env (as every `wsbench judge` child
#    of `wsbench run` is), writes one ledger row from a fake OpenRouter response through llm._chat_result,
#    and with $WSB_OR_STOP present llm._openrouter_once raises JudgeConfigError before touching the client.
# 2. judge.sh watchdog: a copy of judge.sh whose wsbench is a fake (writes $0.40 ledger rows every second,
#    keeps a child process alive, ignores STOP) and whose key check is skipped. Expect: STOP created once the
#    ledger sum passes MAX_USD=2, the fake's whole process group killed ~30 s later, judge exit != 0.
set -uo pipefail
source /workspace/o-lens-improvement/scripts/wsb/env.sh
T=$R/runs/wsb/test_guards; rm -rf "$T"; mkdir -p "$T"
echo "== 1. ledger hook"
env -u OPENROUTER_API_KEY WSB_OR_LEDGER=$T/hook_ledger.jsonl WSB_OR_STOP=$T/HOOK_STOP "$WPY" - <<'EOF'
import asyncio, json, os, subprocess, sys
from types import SimpleNamespace as NS
from wsbench import llm
q = llm._openrouter_once.__qualname__
print("hook in this interpreter:", q, "| _chat_result:", llm._chat_result.__qualname__)
assert "_install" in q, "hook NOT installed"
child = subprocess.run([sys.executable, "-c", "from wsbench import llm; print(llm._openrouter_once.__qualname__)"],
                       capture_output=True, text=True, env=os.environ).stdout.strip()
print("hook in a child interpreter:", child); assert "_install" in child
u = NS(prompt_tokens=1234, completion_tokens=56, cost=0.00123,
       prompt_tokens_details=NS(cached_tokens=1000), completion_tokens_details=NS(reasoning_tokens=40))
resp = NS(usage=u, choices=[NS(finish_reason="stop", message=NS(content='{"a": 1}', refusal=None))])
sp = llm.Spend()
out = llm._chat_result(resp, model="openai/gpt-6-luna", max_tokens=100, spend=sp)
print("parsed:", out, "| Spend.usd:", sp.usd)
rows = [json.loads(l) for l in open(os.environ["WSB_OR_LEDGER"])]
print("ledger rows:", rows); assert len(rows) == 1 and abs(rows[0]["cost"] - 0.00123) < 1e-9
open(os.environ["WSB_OR_STOP"], "w").close()
class Boom:
    def __getattr__(self, k): raise AssertionError("client touched despite STOP")
try:
    asyncio.run(llm._openrouter_once(Boom(), "s", "u", schema={}, model="openai/gpt-6-luna", reasoning=None,
                                     temperature=None, max_tokens=10, timeout=5, spend=sp))
    print("STOP guard: FAILED (no exception)"); sys.exit(1)
except llm.JudgeConfigError as e:
    print("STOP guard ok: JudgeConfigError:", e)
EOF
echo "hook test exit=$?"

echo "== 2. judge.sh watchdog with a fake wsbench"
FAKE=$T/fake_wsbench.sh
cat > "$FAKE" <<'EOF'
#!/bin/bash
# fake wsbench: `report` returns at once; anything else spends $0.40/s into the hook's ledger, with a child
case "$1" in report) echo "fake report"; exit 0;; esac
sleep 1000 &   # a child: must die with the process group
while true; do
  echo "{\"t\": $(date +%s.%N), \"pid\": $$, \"fam\": \"fake\", \"model\": \"x\", \"in\": 1, \"out\": 1, \"cost\": 0.4}" >> "$WSB_OR_LEDGER"
  [ -e "$WSB_OR_STOP" ] && echo "fake: STOP seen at $(date +%T), refusing new calls (keeps running)"
  sleep 1
done
EOF
chmod +x "$FAKE" 2>/dev/null || true
J=$T/judge_test.sh
sed -e "s#\"\$WSBENCH\"#bash $FAKE#g" -e 's#if \[ "\$DRY" != 1 \]; then#if false; then#' "$WSB/judge.sh" > "$J"
mkdir -p "$RO_ROOT/_fake"
for f in poetry introspection; do echo '{"layers": [24, 32], "sampling": "tokens10", "reads": "full"}' > "$RO_ROOT/_fake/$f.jsonl.meta.json"; done
t0=$(date +%s)
LENS=_fake FAMILIES=poetry,introspection MAX_USD=2 OUT=$T/out timeout 120 bash "$J"; rc=$?
echo "judge_test exit=$rc after $(( $(date +%s) - t0 )) s; STOP file: $(ls $T/out/STOP 2>/dev/null || echo none)"
echo "ledger sum: $(python3 $WSB/cost.py --total $T/out)  rows: $(wc -l < $T/out/or_ledger.jsonl)"
sleep 2; left=$(pgrep -f "$FAKE" | wc -l); echo "fake processes left: $left"; pgrep -f "sleep 1000" >/dev/null && echo "orphan sleep children left: $(pgrep -f 'sleep 1000' | wc -l)" || echo "no orphan children"
pkill -f "$FAKE" 2>/dev/null; pkill -f "sleep 1000" 2>/dev/null
rm -rf "$RO_ROOT/_fake"
