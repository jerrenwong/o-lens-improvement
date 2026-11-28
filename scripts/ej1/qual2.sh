cd /workspace/o-lens-improvement
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens ie S1T3" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_ie/E_S1T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "c1 no-suffix S1T3" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_none/E_S1T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens namely S2T3" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_namely/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens thatis S2T3" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_thatis/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens pool3 S2T3" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_ie/E_S2T3.pt,.hf27/ej1/2b_namely/E_S2T3.pt,.hf27/ej1/2b_thatis/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "JPP + elicited ie S2T3" EJ1_MODE=JPP EJ1_E=.hf27/ej1/2b_ie/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "J-lens + elicited ie S2T3" EJ1_MODE=J EJ1_E=.hf27/ej1/2b_ie/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "JPP + c1 S2T3" EJ1_MODE=JPP EJ1_E=.hf27/ej1/2b_none/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "JPP + elicited ie S2T3 + A-lens CE 98k" EJ1_MODE=JPPA EJ1_E=.hf27/ej1/2b_ie/E_S2T3.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens ie S2T3 prior-removed" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_ie/E_S2T3.pt EJ1_PRIOR=.hf27/ej1/2b_ie/prior.pt:S2T3
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "Elicited J-lens thatis S1T1" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_thatis/E_S1T1.pt
bash scripts/qual/dump.sh 2b auto/candidates/ej1_lens.py "c1 no-suffix S1T1" EJ1_MODE=alone EJ1_E=.hf27/ej1/2b_none/E_S1T1.pt
