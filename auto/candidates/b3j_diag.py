"""b3j: diagnostics (label recurrence, leave-one-language-out CV) + full b2e fit ridge (reproduction)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b3j_common", Path(__file__).with_name("b3j_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    _lib.diagnose(ctx)
    return _lib.build_ablation(ctx, "full")
