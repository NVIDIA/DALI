#!/bin/bash -ex

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    # use `which` to invoke test binary with full path so
    # https://google.github.io/googletest/advanced.html#death-test-styles which runs tests in
    # a separate process don't use PATH to discover the file location and fails
    # PackedBFrames test is disabled because it doesn't work with the conda upstream build
    # of FFMpeg
    $(which $BINNAME) --gtest_filter="*:-*PackedBFrames*"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
