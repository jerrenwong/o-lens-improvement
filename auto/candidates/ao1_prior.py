"""ao1 candidate: ask-only lens, mode prior (see ao1_lib.py; env AO1_J, AO1_E[, AO1_E2])."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import build_prior as build  # noqa: E402,F401
