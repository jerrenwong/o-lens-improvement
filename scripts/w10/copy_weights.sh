#!/bin/bash
# copy Qwen3.6-27B weights from /workspace (symlinked blobs) to pod-local /dev/shm/hf27 (w8lib default HF_HOME)
set -e
S=/dev/shm/hf27/hub/models--Qwen--Qwen3.6-27B
SRC=/root/hf27/hub/models--Qwen--Qwen3.6-27B
mkdir -p $S/refs $S/snapshots
cp $SRC/refs/main $S/refs/main
time cp -rL $SRC/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5 $S/snapshots/
du -sh $S
