"""b3j: b2e fit ridge with language-name intermediates removed from targets"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b3j_common", Path(__file__).with_name("b3j_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_ablation(ctx, "nolang")
