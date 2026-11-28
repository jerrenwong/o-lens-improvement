# phase-1 evaluation of the 9B 256-passage fits (thatis, c1); waits for both fits' final E
cd /workspace/o-lens-improvement
until [ -f .hf27/ej1/9b256/thatis/info.json ] && [ -f .hf27/ej1/9b256/none/info.json ]; do sleep 60; done
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited thatis S1T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited thatis S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B c1 S1T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/none/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B c1 S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/none/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + elicited thatis S2T3 n256" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + elicited thatis S1T3 n256" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b256/thatis/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + A-lens U CE 98k + elicited thatis S2T3 n256" EJ1_MODE=JPPA EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt ALENS_E=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt ALENS_E2=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + A-lens CE 98k + elicited thatis S2T3 n256" EJ1_MODE=JPPA EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + c1 S2T3 n256" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b256/none/E_S2T3.pt
for n in 64 128 192; do for v in S1T3 S2T3; do
  bash scripts/ej1/dev.sh 9b ej1_9b256_thatis_${v}_n$n EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/n$n/E_$v.pt
  bash scripts/ej1/dev.sh 9b ej1_9b256_none_${v}_n$n EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/none/n$n/E_$v.pt
done; done
