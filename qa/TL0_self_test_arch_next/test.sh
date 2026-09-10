#!/bin/bash -ex

pip_packages='${python_test_runner_package} numpy opencv-python-headless nvidia-ml-py==11.450.51 numba pillow'

target_dir=./dali/test/python

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    FULLPATH="$(find_test_bin "$BINNAME")"

    "$FULLPATH" --gtest_filter="HwDecoder*"
  done

  # test decoders on A100 as well
  ${python_new_invoke_test} -s decoder test_image

  # test Optical Flow
  ${python_new_invoke_test} -s operator_1 test_optical_flow
  ${python_new_invoke_test} -s checkpointing test_dali_stateless_operators.test_optical_flow_stateless
  ${python_new_invoke_test} test_dali_variable_batch_size.test_optical_flow
}

pushd ../..
source ./qa/test_template.sh
popd
