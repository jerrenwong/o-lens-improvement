cd /workspace/o-lens-improvement
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + elicited JPP-rules thatis S1T3 n256" EJ1_MODE=JPP EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S1T3.pt
bash scripts/qual/dump.sh 9b auto/candidates/ej1_lens.py "9B JPP + A-lens U CE 98k + elicited JPP-rules thatis S1T3 n256" EJ1_MODE=JPPA EJ1_E=.hf27/ej1/9b256_rlens/thatis/E_S1T3.pt ALENS_E=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt ALENS_E2=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt
