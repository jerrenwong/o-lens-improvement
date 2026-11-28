#!/bin/bash
# q2: run a queue of qual dumps one after another (ON THE POD, inside a scripts/q1/job.sh session).
#   bash scripts/q2/q2_queue.sh <queue file>
# Queue line: <9b|2b>|<candidate.py>|<lens name>|VAR=value VAR=value ...   (# comments, blank lines skipped)
# Before each dump: wait until GPU 0 has >= Q2_FREE_MB (default 30000) MiB free. A dump whose json exists is skipped.
cd /workspace/o-lens-improvement
need=${Q2_FREE_MB:-30000}
while IFS='|' read -r tag cand name vars; do
  [ -z "$tag" ] && continue; [[ $tag == \#* ]] && continue
  slug=$(python3 -c 'import sys,re;s=sys.argv[1].replace("++","pp").replace("+","_").lower();print(re.sub(r"[^a-z0-9]+","_",s).strip("_"))' "$name")
  if [ -f "runs/qual/$tag/$slug.json" ]; then echo "skip $tag $name (exists)"; continue; fi
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)" -ge "$need" ]; do sleep 30; done
  echo "=== $(date +%T) $tag | $name | $vars"
  # shellcheck disable=SC2086
  bash scripts/qual/dump.sh "$tag" "$cand" "$name" $vars < /dev/null 2>&1 | grep -v "Loading weights" | tail -25
  echo "=== $(date +%T) done $name"
done < "$1"
