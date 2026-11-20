"""b1b: J++ unit readout minus 1.0 x mean lens vector over period/newline positions"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b1b_lib", Path(__file__).with_name("b1b_lib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "bnd", alpha=1.0)
