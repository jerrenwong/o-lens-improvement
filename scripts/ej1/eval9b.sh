cd /workspace/o-lens-improvement
until [ -f .hf27/ej1/9b256/ie/info.json ] && [ -f .hf27/ej1/9b256_rlens/thatis/info.json ]; do sleep 60; done
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited JPP-rules thatis S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited JPP-rules thatis S1T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited ie S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/ie/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited ie S1T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/ie/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited pool thatis+ie S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt,.hf27/ej1/9b256/ie/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + elicited JPP-rules thatis S2T3 n256" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S2T3.pt
export QUAL_PROBES=probes/probes_v2.json QUAL_SETS=probes QUAL_OUTDIR=runs/qual/9b_v2
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited JPP-rules thatis S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited pool thatis+ie S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt,.hf27/ej1/9b256/ie/E_S2T3.pt
