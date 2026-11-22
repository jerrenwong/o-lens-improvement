"""ja1: a J_ask lens alone (env JA1_J = dict {layer: J}); h @ J^T."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ja1_lib import build_alone as build  # noqa: E402,F401
