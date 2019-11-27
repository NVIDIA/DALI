#!/bin/bash -e

# used pip packages
pip_packages="jupyter matplotlib mxnet-cu{cuda_v} tensorflow-gpu torchvision torch paddle"
target_dir=./docs/examples

do_once() {
  # We need cmake to run the custom plugin notebook + ffmpeg and wget for video example
  apt-get update
  apt-get install -y --no-install-recommends wget ffmpeg cmake
  mkdir -p idx_files
}

test_body() {
  # attempt to run jupyter on all example notebooks
  black_list_files="optical_flow_example.ipynb\|tensorflow-dataset-*\|#" 
  # optical flow requires TU102 architecture whilst currently L1_jupyter_plugins test can be run only on V100
  # tensorflow-dataset requires TF >= 1.15, they are run in TL1_tensorflow_dataset


  # test code
  find */* -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
  python${PYVER:0:1} pytorch/resnet50/main.py -t
}

pushd ../..
source ./qa/test_template.sh
popd
