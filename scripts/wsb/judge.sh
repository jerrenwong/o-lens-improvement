#!/bin/bash
# Judge one lens's readouts with WorkspaceBench (summary mode, fast subset) through OpenRouter. Resumable.
# Usage (pod, after setup_pod.sh and make_readouts.py; the key comes ONLY from the caller's environment):
#   LENS=jpp bash judge.sh                                   # all 18 fast families, Luna recipe, MAX_USD=25
#   LENS=jpp FAMILIES=poetry,model_multihop bash judge.sh    # some families
#   LENS=jpp FAMILIES=poetry LIMIT=5 bash judge.sh           # first 5 items (bank order) of each family: trial
#   LENS=jpp JUDGE=anthropic/claude-haiku-5.5 bash judge.sh  # another OpenRouter judge (no Luna recipe extras)
#   LENS=jpp DRY=1 bash judge.sh                             # dry_run=True: checks files and prompts, no call
# Env: LENS (dir under $RO_ROOT), JUDGE (default openai/gpt-6-luna), FAMILIES (default: the 18 fast families),
#   LIMIT, LAYERS (default: the sidecar's layers), MAX_USD (default 25, cumulative per output dir),
#   CONCURRENCY (calls in flight per family, default 64), SLOTS (cap over all processes, default 256),
#   FAMILY_WORKERS (default 8), OUT (default $JUDGED_ROOT/<judge tag>/<LENS>[_L<n>][_lim<N>]), EXTRA_OPTS.
# What it runs (docs/judges.md "The validated Luna instrument is a recipe"; OpenRouter route instead of their
# OpenAI-direct route: ASSUMPTION that Luna on OpenRouter behaves like Luna on OpenAI):
#   WSBENCH_OPENROUTER_MAX_OUT=32000 wsbench run --fast families=<A> judge_model=$JUDGE judge_effort=low
#       opts=fallback=none,summ_context=1050000,summ_store=<dir>
#   and hinted_cutoff / introspection / model_multihop (group B) with judge_effort=default, as separate
#   `wsbench judge family=F --fast` jobs into the same out dir (a second `run` would rewrite run.json),
#   then `wsbench report dir=$OUT --fast`. Both groups run at once, sharing WSB_SLOTS.
#   opts=fallback=none: no cross-route (Claude) fallback, required with an OpenRouter key only. ANTHROPIC_* is
#   unset for every wsbench process, so no Claude call can be made.
# Spend: wsbench writes $WSB_TOKEN_LEDGER rows for Anthropic calls only (llm.py `ledger`), so the hook
#   wsb_or_ledger.py writes $OUT/or_ledger.jsonl (OpenRouter's own `usage.cost` per call). A watchdog sums it
#   every 5 s; past MAX_USD it creates $OUT/STOP (the hook then refuses every new call) and kills the run 30 s
#   later. The tool itself has no spend cap. A hard cap independent of this script: give the run an
#   OpenRouter key with a credit limit (set in the OpenRouter dashboard).
# Resume: run the same command again (same OUT): cells.jsonl caches every digest and verdict; only missing or
#   failed calls are sent. Delete $OUT/STOP first if the cap stopped it (and raise MAX_USD).
set -uo pipefail
source /workspace/o-lens-improvement/scripts/wsb/env.sh
: "${LENS:?set LENS=<readouts dir name under $RO_ROOT>}"
JUDGE=${JUDGE:-openai/gpt-6-luna}
FAMILIES=${FAMILIES:-$FAST_FAMILIES}
LIMIT=${LIMIT:-0}
MAX_USD=${MAX_USD:-25}
CONCURRENCY=${CONCURRENCY:-64}
SLOTS=${SLOTS:-256}
FAMILY_WORKERS=${FAMILY_WORKERS:-8}
DRY=${DRY:-0}
EXTRA_OPTS=${EXTRA_OPTS:-}
RO=$RO_ROOT/$LENS
[ -d "$RO" ] || { echo "no readouts dir $RO"; exit 2; }
if [ "$DRY" != 1 ]; then
  case "${OPENROUTER_API_KEY:-}" in sk-or-*) ;; *) echo "OPENROUTER_API_KEY missing or not an sk-or- key (not printed)"; exit 2;; esac
fi
first=${FAMILIES%%,*}
LAYERS=${LAYERS:-$("$WPY" -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['layers'])))" "$RO/$first.jsonl.meta.json")}
NL=$(echo "$LAYERS" | tr ',' '\n' | wc -l)
TAG=$(echo "$JUDGE" | tr '/' '_')
SUF=""; [ "$NL" != 11 ] && SUF="_L$NL"; [ "$LIMIT" != 0 ] && SUF="${SUF}_lim$LIMIT"
OUT=${OUT:-$JUDGED_ROOT/$TAG/$LENS$SUF}
mkdir -p "$OUT"
LOG=$OUT/judge_$(date +%Y%m%d_%H%M%S).log
# per-layer digests are shared between layer configs of the same lens and judge (summ_store, docs/summarizer.md)
SUMM_STORE=$R/.hf27/wsb/summ_store/$TAG/$LENS
mkdir -p "$SUMM_STORE"

IS_LUNA=0; [ "$JUDGE" = openai/gpt-6-luna ] && IS_LUNA=1
OPTS="fallback=none,summ_store=$SUMM_STORE"
[ "$IS_LUNA" = 1 ] && OPTS="$OPTS,summ_context=1050000"
[ -n "$EXTRA_OPTS" ] && OPTS="$OPTS,$EXTRA_OPTS"
B_FAMS="hinted_cutoff introspection model_multihop"   # judge_effort=default in the validated recipe
A_LIST=""; B_LIST=""
for f in ${FAMILIES//,/ }; do
  [ -f "$RO/$f.jsonl.meta.json" ] || { echo "skip $f: no readouts/sidecar in $RO"; continue; }
  if [ "$IS_LUNA" = 1 ] && [[ " $B_FAMS " == *" $f "* ]]; then B_LIST="$B_LIST $f"; else A_LIST="$A_LIST,$f"; fi
done
A_LIST=${A_LIST#,}

COMMON=(--fast judge_model="$JUDGE" layers="$LAYERS" concurrency="$CONCURRENCY" opts="$OPTS")
[ "$LIMIT" != 0 ] && COMMON+=(limit="$LIMIT" allow_missing=True)
[ "$DRY" = 1 ] && COMMON+=(dry_run=True)
EFF_A=(); [ "$IS_LUNA" = 1 ] && EFF_A=(judge_effort=low)
EFF_B=(judge_effort=default)

export WSB_OR_LEDGER=$OUT/or_ledger.jsonl WSB_OR_STOP=$OUT/STOP
export WSB_TOKEN_LEDGER=$OUT/anthropic_ledger.jsonl        # stays empty: proves no Claude call was made
# FIX: honour a caller's WSB_SLOTS so several judge.sh (arms) in parallel share one cap
export WSB_SLOTS=${WSB_SLOTS:-/tmp/wsb_slots_$$:$SLOTS}
[ "$IS_LUNA" = 1 ] && export WSBENCH_OPENROUTER_MAX_OUT=32000
unset WSBENCH_OPENAI_ROUTE
[ -e "$OUT/STOP" ] && { echo "$OUT/STOP exists (spend cap hit earlier): remove it to continue"; exit 2; }
{
  echo "== $(date -Is) lens=$LENS judge=$JUDGE layers=$LAYERS limit=$LIMIT max_usd=$MAX_USD"
  echo "   group A (run): $A_LIST"; echo "   group B (judge, effort default): $B_LIST"; echo "   out=$OUT"
} | tee -a "$LOG"

cd "$WSB_SRC" || exit 2
NOCLAUDE=(env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_PROFILE)
PIDS=()
if [ -n "$A_LIST" ]; then   # setsid: each job is its own process group, so the watchdog can kill its children
  setsid "${NOCLAUDE[@]}" "$WSBENCH" run families="$A_LIST" readouts_root="$RO" out="$OUT" \
    family_workers="$FAMILY_WORKERS" "${COMMON[@]}" "${EFF_A[@]}" >> "$LOG" 2>&1 &
  PIDS+=($!)
fi
for f in $B_LIST; do
  mkdir -p "$OUT/$f"
  setsid "${NOCLAUDE[@]}" "$WSBENCH" judge family="$f" readouts="$RO/$f.jsonl" out="$OUT/$f" \
    "${COMMON[@]}" "${EFF_B[@]}" > "$OUT/judge_$f.log" 2>&1 &
  PIDS+=($!)
done

alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
# watchdog: OpenRouter spend from the hook's ledger (cost.py --total sums `cost` over $OUT/or_ledger.jsonl)
(
  while alive; do
    sleep 5
    usd=$(python3 "$WSB/cost.py" --total "$OUT" 2>/dev/null || echo 0)
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$usd" "$MAX_USD"; then
      echo "== $(date -Is) spend \$$usd > MAX_USD \$$MAX_USD: STOP" | tee -a "$LOG"
      touch "$OUT/STOP"; sleep 30
      for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
      break
    fi
  done
) &
WD_PID=$!
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=$?; done
kill "$WD_PID" 2>/dev/null
echo "== $(date -Is) judge exit=$RC" | tee -a "$LOG"
# the table over every family in OUT (rewrites summary.md; Luna rows are never pinned, so the tool's macro is empty)
"$WSBENCH" report dir="$OUT" --fast >> "$LOG" 2>&1
python3 "$WSB/cost.py" "$OUT" | tee "$OUT/cost.md"
exit $RC
