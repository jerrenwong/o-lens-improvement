"""ao1 candidate: ask-only lens, mode three (see ao1_lib.py; env AO1_J, AO1_E[, AO1_E2])."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import build_three as build  # noqa: E402,F401
