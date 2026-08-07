#!/bin/bash -e

# used pip packages
pip_packages='jupyter matplotlib paddlepaddle-gpu'
target_dir=./docs/examples/

do_once() {
  mkdir -p idx_files
}

test_body() {
  # dummy
  exclude_files="#"

  # test code
  find frameworks/paddle -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600 {}
}

pushd ../..
source ./qa/test_template.sh
popd
