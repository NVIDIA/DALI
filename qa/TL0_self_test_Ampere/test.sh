#!/bin/bash -ex

pip_packages='${python_test_runner_package} numpy opencv-python-headless nvidia-ml-py==11.450.51 numba'

target_dir=./dali/test/python

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    for DIRNAME in \
      "../../build/dali/python/nvidia/dali" \
      "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
    do
        if [ -x "$DIRNAME/test/$BINNAME" ]; then
            FULLPATH="$DIRNAME/test/$BINNAME"
            break
        fi
    done

    if [[ -z "$FULLPATH" ]]; then
        echo "ERROR: $BINNAME not found"
        exit 1
    fi

    "$FULLPATH" --gtest_filter="HwDecoder*"
  done

  # test decoders on A100 as well
  ${python_new_invoke_test} -s decoder test_image

  # test Optical Flow
  ${python_new_invoke_test} -s operator_1 test_optical_flow
  ${python_new_invoke_test} -s checkpointing test_dali_stateless_operators.test_optical_flow_stateless
  ${python_invoke_test} test_dali_variable_batch_size.py:test_optical_flow
}

pushd ../..
source ./qa/test_template.sh
popd
