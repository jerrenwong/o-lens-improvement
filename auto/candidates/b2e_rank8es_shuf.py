"""b2e: control: rank8es with shuffled training labels (held-out early stop on true labels)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2e_common", Path(__file__).with_name("b2e_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "rank", src="gen", r=8, es=True, shuffle=True)
