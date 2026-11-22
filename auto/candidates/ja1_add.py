"""ja1: unit(J++ h) + lam unit(J_ask h) (env JA1_J), lam chosen on FIT by max pass@1 + pass@10 (fm27_F4 rule)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ja1_lib import build_added as build  # noqa: E402,F401
