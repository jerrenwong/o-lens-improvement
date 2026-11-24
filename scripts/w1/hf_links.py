"""w1 (pod only): the .hf27 snapshots on /workspace are 0-byte files (geesefs keeps no symlinks), so HF cannot load
Qwen3.6-27B / Olmo / Gemma from it. This builds a symlinked HF_HOME on the POD's local disk (/root/w1_hf) whose
snapshot files point at the existing blobs in .hf27/hub/<repo>/blobs. Name -> blob comes from the Hub's file tree
(metadata call only, no weights downloaded): LFS files by sha256, other files by git blob id.
    python scripts/w1/hf_links.py Qwen/Qwen3.6-27B allenai/Olmo-3-7B-Instruct google/gemma-4-E4B-it
Then run evaluators with HF_HOME=/root/w1_hf."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SRC = Path("/workspace/o-lens-improvement/.hf27/hub")
DST = Path("/root/w1_hf/hub")

api = HfApi()
for repo in sys.argv[1:]:
    d = "models--" + repo.replace("/", "--")
    rev = (SRC / d / "refs" / "main").read_text().strip()
    blobs = SRC / d / "blobs"
    have = set(os.listdir(blobs))
    out = DST / d
    (out / "refs").mkdir(parents=True, exist_ok=True)
    (out / "refs" / "main").write_text(rev)
    snap = out / "snapshots" / rev
    names = sorted(os.listdir(SRC / d / "snapshots" / rev))
    tree = {f.path: f for f in api.list_repo_tree(repo, revision=rev, recursive=True) if hasattr(f, "blob_id")}
    missing = []
    for n in names:
        f = tree.get(n)
        bid = (f.lfs.sha256 if f is not None and f.lfs else f.blob_id) if f is not None else None
        if bid is None or bid not in have:
            missing.append(n); continue
        p = snap / n
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(blobs / bid)
    print(repo, rev, f"{len(names) - len(missing)}/{len(names)} linked", "missing:", missing, flush=True)
