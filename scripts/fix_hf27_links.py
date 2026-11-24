"""Rebuild a usable HF cache for Qwen3.6-27B on the pod's LOCAL disk: /workspace keeps the blobs but not the
snapshot symlinks. Creates /root/hf27/hub/models--Qwen--Qwen3.6-27B/{refs,snapshots} with symlinks into the
blobs on /workspace. Use with HF_HOME=/root/hf27 HF_HUB_OFFLINE=1."""
import os, sys
from huggingface_hub import HfApi
REPO = "Qwen/Qwen3.6-27B"
SRC = "/workspace/o-lens-improvement/.hf27/hub/models--Qwen--Qwen3.6-27B"
DST = "/root/hf27/hub/models--Qwen--Qwen3.6-27B"
rev = open(f"{SRC}/refs/main").read().strip()
info = HfApi().model_info(REPO, revision=rev, files_metadata=True)
os.makedirs(f"{DST}/refs", exist_ok=True); open(f"{DST}/refs/main", "w").write(rev)
snap = f"{DST}/snapshots/{rev}"; os.makedirs(snap, exist_ok=True)
missing = 0
for s in info.siblings:
    blob = s.lfs.sha256 if s.lfs else s.blob_id
    b = f"{SRC}/blobs/{blob}"
    if not os.path.exists(b) or (s.size and os.path.getsize(b) != s.size):
        print("MISSING/size mismatch", s.rfilename, blob, flush=True); missing += 1; continue
    dst = f"{snap}/{s.rfilename}"; os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(b, dst)
print("rev", rev, "files", len(info.siblings), "missing", missing)
sys.exit(1 if missing else 0)
