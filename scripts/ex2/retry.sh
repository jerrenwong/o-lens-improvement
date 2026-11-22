#!/bin/bash
# retry a GPU command until it exits 0 (other jobs share the GPU; OOM is expected); max 60 tries, 120 s apart
log=$1; shift
for i in $(seq 1 60); do
  "$@" >> "$log" 2>&1 && { echo "[retry] ok after $i" >> "$log"; exit 0; }
  echo "[retry] attempt $i failed $(date -u +%H:%M:%S)" >> "$log"; sleep 120
done
exit 1
