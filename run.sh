#!/bin/bash

set -e

#===== begin config =======

stage=0
fix_scp=false
show_vpc_scores=false

anon_exp_parameter="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker"

original_dset=xvect_libri_test_enrolls_anon
# ONE OF: xvect_libri_test_trials_f xvect_libri_test_trials_m xvect_libri_test_enrolls

anon_dset=xvect_libri_test_enrolls

#=====  end config  =======
. utils/parse_options.sh || exit 1;
. ./env.sh

if [ $stage -le -1 ]; then
  printf "${GREEN}Stage -1: testing x-vector loader${NC}\n"

  for anon_exp_parameter in \
  "x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker" \
  "x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker_retrained_xtractor" \
  "x_vector_vpc__crossgender=false__f0transformation=true__diffpseudospeaker" \
  "x_vector_vpc__crossgender=false__f0transformation=true__diffpseudospeaker_retrained_xtractor" \
  "x_vector_vpc__crossgender=true__f0transformation=false__diffpseudospeaker" \
  "x_vector_vpc__crossgender=true__f0transformation=false__diffpseudospeaker_retrained_xtractor" \
  "x_vector_vpc__crossgender=true__f0transformation=true__diffpseudospeaker" \
  "x_vector_vpc__crossgender=true__f0transformation=true__diffpseudospeaker_retrained_xtractor" \
  ;do

    printf "$anon_exp_parameter\n"

    if $fix_scp; then
      cd "data/$anon_exp_parameter"
      rg "exp/models/asv_eval[^/]*/xvect_01709_1" --files-with-matches | \
        xargs sed -i "s|exp/models/asv_eval[^/]*/xvect_01709_1|data/$anon_exp_parameter|g" || true
      cd -
    fi

    for suffix in "" "_anon"; do
      for original_dset in xvect_libri_test_trials_f xvect_libri_test_trials_m xvect_libri_test_enrolls; do
        original_dset=${original_dset}${suffix}
        printf "  $original_dset\n"
        python ./align.py \
           ./data/$anon_exp_parameter/$original_dset/xvector.scp \
           ./data/$anon_exp_parameter/$anon_dset/xvector.scp --test > /dev/null
      done
    done
  done
  printf "${GREEN}Stage -1: All Scp files could be read${NC}\n"
  exit 0
fi

if [ $stage -le 0 ]; then
  printf "$original_dset\n"
  printf "$anon_dset\n"

  if $show_vpc_scores; then
    printf "${RED}Spk verif scores:${NC}\n"
    cat ./data/$anon_exp_parameter/results/results.txt | grep ".*$(echo $original_dset | sed -e 's/xvect_//').*" -A 3
    printf "${RED}with retrained x-vector:${NC}\n"
    cat ./data/${anon_exp_parameter}_retrained_xtractor/results/results.txt | grep ".*$(echo $original_dset | sed -e 's/xvect_//').*" -A 3
    printf "${RED}---${NC}\n"
  fi
  echo ./data/$anon_exp_parameter/$anon_dset/xvector.scp
  echo ./data/${anon_exp_parameter}_retrained_xtractor/$original_dset/xvector.scp

  mkdir -p numpy_arrays
  python ./align.py \
     ./data/$anon_exp_parameter/$anon_dset/xvector.scp \
     ./data/${anon_exp_parameter}_retrained_xtractor/$original_dset/xvector.scp \
     --filter_scp_trials_enrolls
fi

if [ $stage -le 1 ]; then
  python ./Wasserstein_Procrustes.py
fi
