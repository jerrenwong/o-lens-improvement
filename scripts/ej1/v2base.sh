cd /workspace/o-lens-improvement
export QUAL_PROBES=probes/probes_v2.json QUAL_SETS=probes QUAL_OUTDIR=runs/qual/9b_v2
bash scripts/qual/dump.sh 9b auto/candidates/alens.py "J-lens" ALENS_BASE=J0
bash scripts/qual/dump.sh 9b auto/candidates/alens.py "J++" ALENS_BASE=JPP0
bash scripts/qual/dump.sh 9b auto/candidates/w9_alens.py "A-lens U CE 98k" ALENS_BASE=none ALENS_E=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt ALENS_E2=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt
bash scripts/qual/dump.sh 9b auto/candidates/w9_alens.py "J++ + A-lens U CE 98k" ALENS_BASE=JPP ALENS_E=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt ALENS_E2=.hf27/w9/maps/E_9b_spec_lin_ce_98k.pt
