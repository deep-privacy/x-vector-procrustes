#!/bin/bash

set -e

#===== begin config =======

stage=0

anon_exp_parameter="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker"
anon_exp_parameter="x_vector_vpc__crossgender=false__f0transformation=true__diffpseudospeaker"

original_dset=xvect_libri_test_trials_f
# ONE OF: xvect_libri_test_trials_f xvect_libri_test_trials_m xvect_libri_test_enrolls

anon_dset=xvect_libri_test_enrolls_anon

#=====  end config  =======
. utils/parse_options.sh || exit 1;
. ./env.sh

if [ $stage -le -1 ]; then
  printf "${GREEN}Stage -1: testing x-vector loader${NC}\n"

  for anon_exp_parameter in \
  "x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker" \
  "x_vector_vpc__crossgender=false__f0transformation=true__diffpseudospeaker" \
  "x_vector_vpc__crossgender=true__f0transformation=false__diffpseudospeaker" \
  "x_vector_vpc__crossgender=true__f0transformation=true__diffpseudospeaker" \
  ;do

  printf "$anon_exp_parameter\n"
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
  exit 1
fi

if [ $stage -le 0 ]; then
  printf "$original_dset\n"
  printf "$anon_dset\n"
  python ./align.py \
     ./data/$anon_exp_parameter/$anon_dset/xvector.scp \
     ./data/$anon_exp_parameter/$original_dset/xvector.scp \
     --filter_scp_trials_enrolls
fi
