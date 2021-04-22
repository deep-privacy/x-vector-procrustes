#!/bin/bash

set -e

#===== begin config =======

stage=0
fix_scp=false
show_vpc_scores=true

anon_exp_parameter="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker"

# Frontend params
# frontend_train="--pca --pca_n_dim 70"

wass_procrustes_param="--niter 512 --bsz 8 --lr 10"  # Hyperparameter found with grid search

#=====  end config  =======
. utils/parse_options.sh || exit 1;
. ./env.sh

frontend_test="$frontend_train --pca_load_path exp/enroll_train_wp"

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
        python ./prep_dset.py \
          ./data/$anon_exp_parameter/$original_dset/xvector.scp \
          ./data/$anon_exp_parameter/$original_dset/xvector.scp  \
          "/tmp/test/Emb_U" "/tmp/test/User_U" \
          "/tmp/test/Emb_L" "/tmp/test/User_L" \
          --test
      done
    done
  done
  printf "${GREEN}Stage -1: All Scp files could be read${NC}\n"
  exit 0
fi

if [ $stage -le 0 ]; then
  anon_dset=xvect_libri_test_enrolls_anon
  original_dset=xvect_libri_test_enrolls
  # ONE OF: xvect_libri_test_trials_f xvect_libri_test_trials_m xvect_libri_test_enrolls

  if $show_vpc_scores; then
    printf "${RED}Spk verif scores:${NC}\n"
    cat ./data/$anon_exp_parameter/results/results.txt | grep ".*$(echo $anon_dset | sed -e 's/xvect_//').*" -A 3
    printf "${RED}with retrained x-vector:${NC}\n"
    cat ./data/${anon_exp_parameter}_retrained_xtractor/results/results.txt | grep ".*$(echo $anon_dset | sed -e 's/xvect_//').*" -A 3
    printf "${RED}---${NC}\n"
  fi

  printf "${GREEN}   DATA prep:\n     - $original_dset \n     - $anon_dset\n == Data used to train procrustes uv ==${NC}\n"

  expe_dir=exp/enroll_train_wp
  mkdir -p $expe_dir

  python ./prep_dset.py \
     ./data/$anon_exp_parameter/$original_dset/xvector.scp \
     ./data/${anon_exp_parameter}_retrained_xtractor/$anon_dset/xvector.scp \
     "$expe_dir/Emb_U" "$expe_dir/User_U" \
     "$expe_dir/Emb_L" "$expe_dir/User_L" \
     --filter_scp_trials_enrolls \
     --noplot

  printf "${GREEN}Done${NC}\n"
fi

if [ $stage -le 1 ]; then
  printf "${GREEN}== Training procrustes UV ==${NC}\n"

  expe_dir=exp/enroll_train_wp

  python ./Wasserstein_Procrustes.py \
    --emb_src $expe_dir/Emb_U.npy \
    --label_src $expe_dir/User_U.npy \
    --emb_tgt $expe_dir/Emb_L.npy \
    --label_tgt $expe_dir/User_L.npy \
    --rotation exp/WP_R.npy \
    $frontend_train $wass_procrustes_param

  printf "${GREEN}Done${NC}\n"
  # exit 0
fi

if [ $stage -le 2 ]; then
  printf "${GREEN}== TEST trained procrustes UV ==${NC}\n"

  expe_dir=exp/trials_test
  mkdir -p $expe_dir

  for dset in enrolls trials_f trials_m; do

    original_dset=xvect_libri_test_${dset}
    anon_dset=xvect_libri_test_${dset}_anon

    python ./prep_dset.py \
       ./data/$anon_exp_parameter/$original_dset/xvector.scp \
       ./data/${anon_exp_parameter}_retrained_xtractor/$anon_dset/xvector.scp \
       "$expe_dir/Emb_U" "$expe_dir/User_U" \
       "$expe_dir/Emb_L" "$expe_dir/User_L" \
       --filter_scp_trials_enrolls \
       --noplot

    python ./Wasserstein_Procrustes.py \
      --emb_src $expe_dir/Emb_U.npy \
      --label_src $expe_dir/User_U.npy \
      --emb_tgt $expe_dir/Emb_L.npy \
      --label_tgt $expe_dir/User_L.npy \
      --rotation exp/WP_R.npy \
      $frontend_test \
      --test

  done

  printf "${GREEN}Done${NC}\n"
fi

if [ $stage -le 3 ]; then
  for exp in "x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker" \
              "x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker_retrained_xtractor"; do
  slug=original
  if [[ $exp == *retrained* ]]; then
    slug=anon
  fi
    for dset in "f" "m";do
      printf "**ASV ($slug): ${RED}test_trials_${dset} ${GREEN}original${NC} <=> ${RED}test_enrolls - ${GREEN}original${RED}${NC}**\n"
      python compute_spk_cosine.py \
        ./data/${exp}/xvect_libri_test_trials_${dset}/meta/trials \
        ./data/${exp}/xvect_libri_test_trials_${dset}/ \
        ./data/${exp}/xvect_libri_test_enrolls/ \
       ./exp/cosine_scores.txt
    done

    for dset in "f" "m";do
      printf "**ASV ($slug): ${RED}test_trials_${dset} ${GREEN}anonymized${NC} <=> ${RED}test_enrolls - ${GREEN}anonymized${RED}${NC}**\n"
      python compute_spk_cosine.py \
        ./data/${exp}/xvect_libri_test_trials_${dset}/meta/trials \
        ./data/${exp}/xvect_libri_test_trials_${dset}_anon/ \
        ./data/${exp}/xvect_libri_test_enrolls_anon/ \
       ./exp/cosine_scores.txt
    done
    printf "\n"
  done
fi

if [ $stage -le 4 ]; then
  for dset in "f" "m";do
    exp="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker_retrained_xtractor"
    python ./apply_procrustes.py \
      --emb_src ./data/${exp}/xvect_libri_test_trials_${dset}_anon/xvector.scp \
      --emb_out ./data/${exp}/xvect_libri_test_trials_${dset}_anon/ \
      --rotation ./exp/WP_R.npy \
      $frontend_test
  done

  for dset in "f" "m";do
    exp_o="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker"
    exp_a="x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker_retrained_xtractor"

    printf "**ASV (anon - after apply_procrustes): ${RED}test_trials_${dset} ${GREEN}anonymized => procrustes${NC} <=> ${RED}test_enrolls - ${GREEN}original${RED}${NC}**\n"
    python compute_spk_cosine.py \
      ./data/${exp_a}/xvect_libri_test_trials_${dset}/meta/trials \
      ./data/${exp_a}/xvect_libri_test_trials_${dset}_anon/ \
      ./data/${exp_o}/xvect_libri_test_enrolls/ \
     ./exp/cosine_scores.txt \
     --trial-scp transformed_xvector.scp
  done
fi
