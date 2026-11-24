"""w3 candidate: Jacobian A-lens, mode alone (see w3_lib.py; env W3_JT, W3_JU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3_lib import build_alone as build  # noqa: E402,F401
