#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python pillow librosa scipy nvidia-ml-py==11.450.51 numba lz4 numba_cuda[cu${DALI_CUDA_MAJOR_VERSION}]'

target_dir=./dali/test/python

test_body() {
  ${python_new_invoke_test} -A '!slow' -s operator_1 test_numba_func
}

pushd ../..
source ./qa/test_template.sh
popd
