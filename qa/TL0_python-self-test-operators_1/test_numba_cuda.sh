#!/bin/bash -e
# used pip packages

# there's no cuXX extra for the latest numba-cuda and numba-cuda requires cuda-bindings
NUMBA_PKG="numba_cuda cuda-bindings==${DALI_CUDA_MAJOR_VERSION}.*"
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python pillow librosa scipy nvidia-ml-py==11.450.51 numba lz4 ${NUMBA_PKG}'

target_dir=./dali/test/python

test_body() {
  ${python_new_invoke_test} -A '!slow' -s operator_1 test_numba_func
}

pushd ../..
source ./qa/test_template.sh
popd
