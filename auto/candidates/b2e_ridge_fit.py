"""b2e: J++ + kernel-ridge label correction (idea 18) on ctx.fit only; lam, b by 5-fold CV on fit"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2e_common", Path(__file__).with_name("b2e_common.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "ridge", src="fit")
