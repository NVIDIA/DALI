#!/bin/bash -e

build_and_check() {
  make -j
  pip install ./dali/python

  # If there are missing symbols, this will fail
  python -c 'import nvidia.dali'

  # There should be some tests to run
  ./dali/python/nvidia/dali/test/dali_operator_test.bin
  ./dali/python/nvidia/dali/test/dali_kernel_test.bin
}

# Example 1: Building Only Random operators AND Slice family kernels
example_1() {
  cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DBUILD_TEST=ON \
        -DBUILD_BENCHMARK=OFF \
        -DCUSTOM_OPERATOR_SRCS_PATTERN="random/*.cc" \
        -DCUSTOM_OPERATOR_SRCS_PATTERN_EXCLUDE="random/noise/*" \
        -DCUSTOM_OPERATOR_TEST_SRCS_PATTERN="*random/*test*.cc" \
        -DCUSTOM_KERNEL_SRCS_PATTERN="*slice*" \
        -DCUSTOM_KERNEL_TEST_SRCS_PATTERN="slice_*test*" \
        ${DALI_DIR}
  build_and_check
}

# Example 2: Building only TFRecord reader
example_2() {
  cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DBUILD_TEST=ON \
        -DBUILD_BENCHMARK=OFF \
        -DCUSTOM_OPERATOR_SRCS_PATTERN="reader/tf*.cc;reader/loader/loader.cc;decoder/cache/*" \
        -DCUSTOM_OPERATOR_TEST_SRCS_PATTERN=" " \
        -DCUSTOM_KERNEL_SRCS_PATTERN=" " \
        -DCUSTOM_KERNEL_TEST_SRCS_PATTERN=" " \
        ${DALI_DIR}
  build_and_check
}


test_body() {
    export DALI_DIR=$PWD
    export TMP_DIR="$(mktemp -d)"
    pushd ${TMP_DIR}
    example_1
    # Not cleaning in purpose (we don't want to recompile dali_core, etc again)
    example_2
    rm -rf ./*
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
