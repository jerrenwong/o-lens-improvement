#!/bin/bash
# W4 launcher (fleet-run job format; freshness check limited to scripts/w4 so other agents' edits don't block).
#   bash scripts/w4/run.sh <job> "<cmd>"     (run on the hub)
set -e
n=wong-olens4; job=$1; c=$2; dir=/workspace/o-lens-improvement; J=/workspace/.fleet/jobs/$n
sums=$(cd $dir && md5sum scripts/w4/*.py scripts/w4/*.sh)
for i in $(seq 1 40); do ssh $n "cd $dir && md5sum --quiet -c -" <<<"$sums" >/dev/null 2>&1 && break; sleep 5; done
ssh $n "tmux has-session -t $job 2>/dev/null" && { echo "job $job already running"; exit 1; }
rm -f $J/$job.exit
printf '#!/bin/bash\n# job %s on %s, started %s\n# dir: %s\n# cmd: %s\ncd %q || exit 99\n{\n%s\n} > %s/%s.log 2>&1\necho $? > %s/%s.exit\n' \
  "$job" "$n" "$(date -u +%FT%RZ)" "$dir" "$c" "$dir" "$c" "$J" "$job" "$J" "$job" > $J/$job.sh
ssh $n "for i in 1 2 3 4 5 6; do [ -s $J/$job.sh ] && break; sleep 2; done; tmux new-session -d -s $job 'bash $J/$job.sh'"
echo "started $job"
