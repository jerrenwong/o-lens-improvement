#!/bin/bash
# push given files from hub scripts/th1 to pod /root/th1 and compare md5
cd /workspace/o-lens-improvement/scripts/th1
fleet ssh wong-olens2 'mkdir -p /root/th1; rm -f /root/th1_gen2.py /root/th1_words.json /root/th1_th1_common.py' >/dev/null
for f in "$@"; do fleet ssh wong-olens2 "cat > /root/th1/$f" < $f; done
md5sum "$@"; fleet ssh wong-olens2 "cd /root/th1; md5sum $*"
