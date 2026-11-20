"""b2d patchscope (idea 6): unit(JPP h) + 0.5 * unit(S), S = final-block output at the last token of
'Text: <h>\nIn one word, this text is about' (h replaces ' ?' at block `layer`). Best mix on the fit scan."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2d_common", Path(__file__).with_name("b2d_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def build(ctx):
    return C.PatchLens(ctx.JPP, C.Patcher(ctx.model, C.PROMPTS["concept_ph"], "replace"), 0.5)
