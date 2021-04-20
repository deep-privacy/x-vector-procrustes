#!/bin/bash

set -e

mkdir -p exp/oar-log/

# WITH GNU parallel
# Cluster names: grvingt grappe
# oarsub -q production -p "cluster='grvingt'" -l nodes=10,walltime=01:00 'sleep 10d'
# oarsub -C $???OAR_JOB_ID????
# oarprint host -P cpuset,host -C+ -F "1/OAR_USER_CPUSET=% oarsh %" | tee cpu-executors
#                                                                                 ::: pca_dim     ::: niter       ::: bsz       ::: lr
# parallel --bar --workdir $PWD --slf cpu-executors ./parallel.sh {1} {2} {3} {4} ::: 70 80 90    ::: 512 416 300 ::: 6 8 10 12 ::: 8 10 15 20

# Without GNU parallel
for pca_dim in 25 30 35 40
do
  for niter in 256 512 300
  do
    for bsz in 4 8 10 12
    do
      for lr in 10 20 30 50
      do
        ./parallel.sh $pca_dim $niter $bsz $lr
      done
    done
  done
done
