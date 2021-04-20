#!/bin/bash

set -e

mkdir -p exp/oar-log/

pca_dim=$1
niter=$2
bsz=$3
lr=$4

log_suffix="pca_dim=${pca_dim}_niter=${niter}_bsz=${bsz}_lr=${lr}"
./run.sh --stage 1 --frontend_train "--pca --pca_n_dim $pca_dim" --wass_procrustes_param "--niter $niter --bsz $bsz --lr $lr" 2> exp/oar-log/${log_suffix}-err-more.log > exp/oar-log/${log_suffix}-out-more.log
echo exp/oar-log/${log_suffix}-out-more.log >> out
tail -n 4 exp/oar-log/${log_suffix}-out-more.log | head -n1 >> out
tail -n 2 out
