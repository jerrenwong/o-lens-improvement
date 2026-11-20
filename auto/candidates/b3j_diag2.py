"""b3j: diagnostics: strict leave-one-item-out on multilingual fit items (exclude items sharing keys); lens = full b2e ridge"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b3j_common", Path(__file__).with_name("b3j_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    _lib.diagnose_keys(ctx)
    return _lib.build_ablation(ctx, "full")
