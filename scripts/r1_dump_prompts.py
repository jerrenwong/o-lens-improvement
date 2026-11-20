"""Dump the exact WikiText prompt pool the jpp_lens recipe uses (jpp_cli.load_fit_prompts) to JSON.
    cd /root/jpp_lens && .venv/bin/python /workspace/o-lens-improvement/scripts/r1_dump_prompts.py OUT.json [N]"""
import json, sys
sys.path.insert(0, "/root/jpp_lens/scripts")
import jpp_cli
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
p = jpp_cli.load_fit_prompts(n)
json.dump(p, open(sys.argv[1], "w"))
print("wrote", len(p), "prompts to", sys.argv[1])
