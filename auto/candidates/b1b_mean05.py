"""b1b: J++ unit readout minus 0.5 x WikiText mean lens vector per layer"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b1b_lib", Path(__file__).with_name("b1b_lib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "mean", alpha=0.5)
