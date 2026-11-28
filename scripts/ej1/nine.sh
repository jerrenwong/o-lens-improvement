cd /workspace/o-lens-improvement
L=/workspace/.fleet/jobs/wong-olens6/ej1_9b_fit.log
until [ $(grep -c "saved n=48" $L) -ge 2 ]; do sleep 30; done
mkdir -p .hf27/ej1/9b_n48; cp -r .hf27/ej1/9b/ie .hf27/ej1/9b/none .hf27/ej1/9b_n48/
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited J-lens ie S1T3 n48" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/ie/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B c1 no-suffix S2T3 n48" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/none/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited J-lens ie S2T3 n48" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/ie/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B c1 no-suffix S1T3 n48" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/none/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited J-lens ie S1T1 n48" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/ie/E_S1T1.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + elicited ie S1T3 n48" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b_n48/ie/E_S1T3.pt
bash scripts/ej1/dev.sh 9b ej1_9b_J0 EJ1_MODE=J0
bash scripts/ej1/dev.sh 9b ej1_9b_JPP0 EJ1_MODE=JPP0
bash scripts/ej1/dev.sh 9b ej1_9b_none_S1T1_alone EJ1_MODE=alone EJ1_E=.hf27/ej1/9b_n48/none/E_S1T1.pt
