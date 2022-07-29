#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy>=1.17 pillow torch numba scipy librosa==0.8.1'

target_dir=./dali/test/python

test_body() {
  # CPU only test, remove CUDA from the search path just in case
  export LD_LIBRARY_PATH=""
  export PATH=${PATH/cuda/}
  ${python_invoke_test} --attr 'pytorch' test_dali_cpu_only.py
}

pushd ../..
source ./qa/test_template.sh
popd
