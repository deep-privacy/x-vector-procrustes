#!/bin/bash

set -e

nj=$(nproc)

home=$PWD

conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh
venv_dir=$PWD/venv

mark=.done-venv
if [ ! -f $mark ]; then
  echo 'Making python virtual environment'
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate

  echo 'Installing conda dependencies'
  # yes | conda install -c conda-forge XXXXXXXX
  touch $mark
fi
echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi" > env.sh

source $venv_dir/bin/activate

mark=.done-python
if [ ! -f $mark ]; then
  pip install kaldiio==2.17.2
  pip install scikit-learn==0.24.1
  pip install matplotlib==3.4.1
  pip install pot==0.7.0
  pip install torch==1.7.1
  pip install pandas==1.0.5
  pip install seaborn==0.10.1
  cd $home
  touch $mark
fi

mark=.done-anonymization_metrics
if [ ! -f $mark ]; then
  cd utils
  git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
  # git checkout 4787d4f
  cd $home
  touch $mark
fi
