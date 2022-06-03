#!/bin/bash -e

# used pip packages
pip_packages='jupyter matplotlib tensorflow-gpu'
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # attempt to run jupyter on all example notebooks
  exclude_files="tensorflow-dataset*\|#"
  # tensorflow-dataset requires TF >= 1.15, they are run in TL1_tensorflow_dataset


  # test code
  find frameworks/tensorflow -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
}

pushd ../..
source ./qa/test_template.sh
popd
