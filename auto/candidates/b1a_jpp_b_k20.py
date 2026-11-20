import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1a_common


def build(ctx):
    return b1a_common.build(ctx, "jpp", "b", k=20)
