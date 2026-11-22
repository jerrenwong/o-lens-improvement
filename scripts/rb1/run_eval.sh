#!/bin/bash
# rb1 dev evals (pod, tmux rb1ev): waits for READY_<tag>, runs up to 3 evaluations in parallel, then the bootstrap.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/rb1; mkdir -p $L; cd $R
VARS="r300_0 r300_1 r300_2 r1000_0 r1000_1 r1000_2 r3000_0 r3000_1 r3000_2 r10000_0 r10000_1 r10000_2 rall hA hB p1000_0 p1000_1 p10000_0 p10000_1"
for TAG in ${TAGS:-9b 27b 2b}; do
  while [ ! -f $L/READY_$TAG ]; do [ -f $L/FAILED ] && exit 1; sleep 60; done
  {
    echo "$TAG st1_jppu rb1_jppu_$TAG"; echo "$TAG st1_B0 rb1_Lstored_$TAG"
    echo "$TAG st1_B0_A rb1_Lstored_A_$TAG"; echo "$TAG st1_p3v_t10 rb1_P3vstored_$TAG"; echo "$TAG st1_p3v_t10_A rb1_P3vstored_A_$TAG"
    for v in $VARS; do echo "$TAG rb1_$v rb1_${v}_$TAG"; echo "$TAG rb1_${v}_A rb1_${v}_A_$TAG"; done
  } | xargs -P 3 -L 1 bash scripts/rb1/one_eval.sh
  $PY scripts/r2_boot.py rb1_hA_$TAG-rb1_jppu_$TAG rb1_hB_$TAG-rb1_jppu_$TAG rb1_hA_$TAG-rb1_hB_$TAG \
     rb1_Lstored_$TAG-rb1_jppu_$TAG rb1_rall_$TAG-rb1_jppu_$TAG rb1_hA_A_$TAG-rb1_hB_A_$TAG > $L/boot_$TAG.log 2>&1
  touch $L/EVDONE_$TAG
done
if [ -f $L/errors ]; then cp $L/errors $L/FAILED; else touch $L/DONE; fi
