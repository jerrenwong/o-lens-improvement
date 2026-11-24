#!/bin/bash
# W10 GPU chain: B (collect + fit, base only) -> C (readouts + null organisms) -> D (questions to organisms)
P=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement
W10_N=10000 W10_BSY=64 $P scripts/w10/collect.py collect || exit 1
W10_N=10000 $P scripts/w10/read_c.py || exit 2
$P scripts/w10/part_d.py || exit 3
