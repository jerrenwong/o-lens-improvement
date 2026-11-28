#!/bin/bash
# dev scoring of one ej1 fit dir (pod): bash scripts/ej1/eval_fit.sh <2b|9b> <fit name e.g. 2b_ie>
tag=$1; f=$2; D=.hf27/ej1/$f
cd /workspace/o-lens-improvement
for v in S1T1 S1T3 S2T1 S2T3; do
  bash scripts/ej1/dev.sh $tag ej1_${f}_${v}_alone EJ1_MODE=alone EJ1_E=$D/E_$v.pt
  bash scripts/ej1/dev.sh $tag ej1_${f}_${v}_alonepr EJ1_MODE=alone EJ1_E=$D/E_$v.pt EJ1_PRIOR=$D/prior.pt:$v
done
for v in S1T1 S1T3 S2T1 S2T3; do
  bash scripts/ej1/dev.sh $tag ej1_${f}_${v}_jpp EJ1_MODE=JPP EJ1_E=$D/E_$v.pt
  bash scripts/ej1/dev.sh $tag ej1_${f}_${v}_j EJ1_MODE=J EJ1_E=$D/E_$v.pt
done
