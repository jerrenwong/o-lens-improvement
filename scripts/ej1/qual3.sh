cd /workspace/o-lens-improvement
until [ -f /workspace/.fleet/jobs/wong-olens6/ej1_qual2.exit ]; do sleep 30; done
bash scripts/qual/dump.sh 9b auto/candidates/w3_f.py "Jacobian A-lens K1 alone" W3_MODE=alone W3_JT=runs/ao1/9b/lens/topK1_n128.pt W3_JU=runs/w3/9b/lens/unsK1_n128.pt
