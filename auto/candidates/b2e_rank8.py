"""b2e: J++ + rank-8 CE correction trained on self-generated concept scenes (idea 7)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2e_common", Path(__file__).with_name("b2e_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "rank", src="gen", r=8)
