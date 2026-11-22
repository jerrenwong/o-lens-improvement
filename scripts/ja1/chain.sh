#!/bin/bash
# ja1 chain for one model:  bash chain.sh 2b|9b fits|post|eval|all
# Sizes: env NBIG (A1/A2 largest, default 1024 for 2b, 256 for 9b), NSMALL (A3/A4/W1/W2, default 256 for 2b, 64 for 9b).
set -uo pipefail
TAG=$1; shift
C=/root/ja1/code; R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
RUN="bash $C/run_v3.sh $TAG"
O=$R/runs/ja1/$TAG; LG=/root/ja1/$TAG; mkdir -p $O/lens $O/agree $LG
case $TAG in
  2b) M=Qwen/Qwen3.5-2B; NA1=${NA1:-1024}; NA2=${NA2:-256}; NSMALL=${NSMALL:-64}; ML="9 12" ;;
  9b) M=Qwen/Qwen3.5-9B; NA1=${NA1:-64}; NA2=${NA2:-64}; NSMALL=${NSMALL:-64}; ML="12 16" ;;
esac
P="$PY $C/ja1_post.py"
ck() { ls -d $O/$1/*/ja1 $O/$1/*/ja1_at* 2>/dev/null | head -1 | sed 's/_at[0-9]*$//'; }
at() {  # run n -> dir of the n-prompt state (snapshot or final)
  local f; f=$(ck $1); [ -z "$f" ] && return 1
  local n; n=$(python3 -c "import json;print(len(json.load(open('$O/$1/prompts.json'))))")
  if [ "$2" = "$n" ]; then [ -f $O/$1/fit_info.json ] && echo $f; else [ -d ${f}_at$2 ] && echo ${f}_at$2; fi; }

fits() {
  [ -f $O/routers/router_unstated_K8.pt ] || $RUN routers || return 1
  for st in ${FITS:-A2 A3 W1 W2 A1 A4 std}; do
    case $st in
      A2) $RUN fit A2 $NA2 rlens topic ;;
      A1) $RUN fit A1 $NA1 none topic ;;
      std) $RUN jstd ;;
      A3) $RUN fit A3 $NSMALL rlens unstated ;;
      A4) $RUN fit A4 $NSMALL none topic --wrong ;;
      W1) $RUN fit W1 $NSMALL rlens topic1 ;;
      W2) $RUN fit W2 $NSMALL rlens topic2 ;;
    esac
  done
}

pooled() {  # name K dirs... [-- minus dirs]
  local name=$1 K=$2; shift 2
  local dirs=() minus=(); local m=0
  for x in "$@"; do if [ "$x" = "--" ]; then m=1; elif [ $m = 0 ]; then dirs+=("$x"); else minus+=("$x"); fi; done
  [ -f $O/lens/$name.pt ] && return 0
  [ ${#dirs[@]} = 0 ] && return 1
  $P pooled --K $K --dir "${dirs[@]}" ${minus:+--minus "${minus[@]}"} --out $O/lens/$name.pt > $O/lens/$name.json 2>>$LG/post.log
}
agree() {  # name a b
  [ -f $O/lens/$2.pt ] && [ -f $O/lens/$3.pt ] || return 0
  [ -f $O/agree/$1.json ] || bash $C/retry.sh 4000 $LG/agree.log $P agree --model $M --a $O/lens/$2.pt --b $O/lens/$3.pt --acts $O/heldout_acts.pt --out $O/agree/$1.json
}
post() {
  [ -f $O/heldout_acts.pt ] || bash $C/retry.sh 12000 $LG/heldout.log $P heldout --model $M --out $O/heldout_acts.pt
  for r in A1 A2 A3 A4 W1 W2; do
    [ -f $O/$r/mass.pt ] && $P mass --dir $O/$r >> $LG/post.log 2>&1
    for n in 64 256 1024; do
      d=$(at $r $n) || continue; h=$(at $r $((n/2))) || continue
      pooled ${r}_n$n 1 $d
      pooled ${r}_n${n}_hA 1 $h
      pooled ${r}_n${n}_hB 1 $d -- $h
      agree ${r}_n${n}_halves ${r}_n${n}_hA ${r}_n${n}_hB
    done
  done
  # standard J-lens recipe on its own prompts (64; halves 32 / 32)
  f=$(ck std 2>/dev/null; ls -d $O/std/*/std 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    pooled STD_n64 1 $f; pooled STD_n64_hA 1 ${f}_at32; pooled STD_n64_hB 1 $f -- ${f}_at32
    agree STD_n64_halves STD_n64_hA STD_n64_hB
  fi
  # three topic wordings as three targets per passage (fixed passages)
  for n in 64 256; do
    a=$(at A2 $n) && b=$(at W1 $n) && c=$(at W2 $n) || continue
    ha=$(at A2 $((n/2))) && hb=$(at W1 $((n/2))) && hc=$(at W2 $((n/2))) || continue
    pooled W012_n$n 1 $a $b $c
    pooled W012_n${n}_hA 1 $ha $hb $hc
    pooled W012_n${n}_hB 1 $a $b $c -- $ha $hb $hc
    agree W012_n${n}_halves W012_n${n}_hA W012_n${n}_hB
    agree W0vsW012_n$n A2_n$n W012_n$n
  done
  agree A1vsA2_n64 A1_n64 A2_n64
  agree A2vsA3_n64 A2_n64 A3_n64
  agree A1vsA4_n64 A1_n64 A4_n64
  # A5: experts (K8, router fitted on passage-token activations) + stock fit-weights on FIT items
  for n in 64 256 1024; do d=$(at A2 $n) && $RUN a5 A5_n$n $d; done
  for n in 64 256; do d=$(at A3 $n) && $RUN a5 A5u_n$n $d; done
  for n in 64 256 1024; do
    d=$(at A2 $n) && h=$(at A2 $((n/2))) && [ -f $O/lens/A5_n${n}_jpp_expert_weights.json ] || continue
    for half in hA hB; do
      [ -f $O/lens/A5_n${n}_$half.pt ] && continue
      if [ $half = hA ]; then $P a5 --dir $h --weights $O/lens/A5_n${n}_jpp_expert_weights.json --out $O/lens/A5_n${n}_hA.pt >> $LG/post.log 2>&1
      else $P a5 --dir $d --minus $h --weights $O/lens/A5_n${n}_jpp_expert_weights.json --out $O/lens/A5_n${n}_hB.pt >> $LG/post.log 2>&1; fi
    done
    agree A5_n${n}_halves A5_n${n}_hA A5_n${n}_hB
  done
  agree A2vsA5_n64 A2_n64 A5_n64
}

ev() {  # name lensfile cand
  local n=$1 f=$2 c=$3
  [ -f $R/auto/runs/$n/summary.json ] && return 0
  (cd $R && JA1_J=$f bash $C/retry.sh ${NEED:-6000} $LG/eval_$n.log $PY auto/evaluate.py --model $M --lens auto/candidates/$c --name $n --lite)
}
evals_base() {
  ev ja1_${TAG}_JPP x jpp.py
  case $TAG in 2b) ev ja1_${TAG}_askmap x m2b_L.py ;; 9b) ev ja1_${TAG}_askmap x fm27_F4.py ;; esac
}
evals() {
  evals_base
  for L in A1_n64 A2_n64 A3_n64 A4_n64 A5_n64 A5u_n64 W012_n64 A1_n256 A2_n256 A5_n256 A3_n256 A5u_n256 W012_n256 \
           A1_n1024 A2_n1024 A5_n1024 STD_n64; do
    [ -f $O/lens/$L.pt ] || continue
    ev ja1_${TAG}_${L}_alone $O/lens/$L.pt ja1_alone.py
    case $L in STD*) ;; *) ev ja1_${TAG}_${L}_add $O/lens/$L.pt ja1_add.py ;; esac
  done
}
qual() {
  local best=$1
  $PY $R/scripts/ja1/ja1_qual.py --model $M --J $O/lens/$best.pt --layers $ML --out $O/qual_$best.txt > $LG/qual.log 2>&1
}
for s in "$@"; do
  case $s in
    all) fits; post; evals ;;
    qual:*) qual ${s#qual:} ;;
    *) $s ;;
  esac
done
