"""b2e: kernel-ridge correction (idea 18) on generated scenes + ctx.fit; lam, b chosen on held-out generated concepts"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2e_common", Path(__file__).with_name("b2e_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "ridge", src="genfit")
