#!/bin/bash
# dp1: run post-processing as each fit finishes, then write /root/dp1/DONE (or FAILED).
OUT=/workspace/o-lens-improvement/runs/dp1
j=0; n=0
while [ $j = 0 ] || [ $n = 0 ]; do
  if [ $j = 0 ] && [ -f /root/dp1/fits2b.done ]; then
    if ls $OUT/2b_jpp/ckpt/*/fit/shard0of1_dp1_stats.json >/dev/null 2>&1; then bash /root/dp1/code/post.sh jpp2b; else echo "jpp2b fit FAILED" >> /root/dp1/status; fi; j=1; fi
  if [ $n = 0 ] && [ -f /root/dp1/fits9b.done ]; then
    if ls $OUT/9b_plain/ckpt/*/fit/shard0of1_dp1_stats.json >/dev/null 2>&1; then bash /root/dp1/code/post.sh plain9b; else echo "plain9b fit FAILED" >> /root/dp1/status; fi; n=1; fi
  sleep 120
done
if grep -q FAILED /root/dp1/status 2>/dev/null; then grep FAILED /root/dp1/status > /root/dp1/FAILED; else date > /root/dp1/DONE; fi
