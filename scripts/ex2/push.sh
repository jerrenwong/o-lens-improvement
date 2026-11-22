#!/bin/bash
# push hub-written ex2 files to the pod over ssh, written in place (hub->pod sync of edits can lag); verify md5
for f in "$@"; do
  p=$(readlink -f "$f")
  a=$(md5sum < "$p")
  b=$(timeout 60 ssh -o ConnectTimeout=40 wong-olens2 "cat > $p; md5sum < $p" < "$p")
  [ "$a" == "$b" ] && echo "ok $p" || echo "MISMATCH $p"
done
