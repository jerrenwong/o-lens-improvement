cd /workspace/o-lens-improvement
export QUAL_PROBES=probes/probes_v2.json QUAL_SETS=probes QUAL_OUTDIR=runs/qual/9b_v2
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited thatis S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S2T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B elicited thatis S1T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/thatis/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B c1 S2T3 n256" EJ1_MODE=alone EJ1_E=.hf27/ej1/9b256/none/E_S2T3.pt
