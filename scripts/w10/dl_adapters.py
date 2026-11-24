"""download the four AuditBench organism adapters to pod-local /dev/shm/w10dl (symlinked from /root/w8/adapters)/<tag> (w8lib.ADIR)."""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_XET_CACHE"] = "/dev/shm/w10dl/xet"
from huggingface_hub import snapshot_download
for q in ["flattery", "secret_loyalty", "hardcode_test_cases", "anti_ai_regulation"]:
    p = snapshot_download("agu18dec/auditbench-qwen3.6-27b-model-organisms", allow_patterns=[f"organisms/{q}_tdkto_r64/*"],
                          local_dir="/dev/shm/w10dl", cache_dir="/dev/shm/w10dl/hfc")
    os.makedirs("/root/w8/adapters", exist_ok=True)
    src = f"/dev/shm/w10dl/organisms/{q}_tdkto_r64"
    dst = f"/root/w8/adapters/{q}_tdkto_r64"
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print("ok", q, flush=True)
