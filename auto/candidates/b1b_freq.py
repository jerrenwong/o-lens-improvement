"""b1b: J++ plus b x ridge logit direction of log unigram frequency (b<=0 chosen on fit)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b1b_lib", Path(__file__).with_name("b1b_lib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
Lens = _lib.Lens


def build(ctx):
    return _lib.build_variant(ctx, "prior", score="logfreq", grid=[0.0, -0.1, -0.2, -0.3, -0.5, -1.0])
