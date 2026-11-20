#!/bin/bash
# x1: dev runs of the context-routed readout candidates, one at a time.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
RUN="bash scripts/x1_run.sh"
C=auto/candidates/x1_route.py
X1_MODE=hard   X1_COND=A               $RUN /root/x1_hardA.log   $PY auto/evaluate.py --lens $C --name x1_hardA
X1_MODE=soft   X1_COND=A X1_PMAX=0.5   $RUN /root/x1_soft05.log  $PY auto/evaluate.py --lens $C --name x1_soft05
X1_MODE=soft   X1_COND=A X1_PMAX=0.8   $RUN /root/x1_soft08.log  $PY auto/evaluate.py --lens $C --name x1_soft08
X1_MODE=mix    X1_COND=A               $RUN /root/x1_mixA.log    $PY auto/evaluate.py --lens $C --name x1_mixA
X1_MODE=hard   X1_COND=C               $RUN /root/x1_hardC.log   $PY auto/evaluate.py --lens $C --name x1_hardC
X1_MODE=pooled X1_COND=A               $RUN /root/x1_pooledA.log $PY auto/evaluate.py --lens $C --name x1_pooledA
touch /root/x1_evals_done
