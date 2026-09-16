#!/bin/bash -ex

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    FULLPATH="$(find_test_bin "$BINNAME")"

    DALI_USE_EXEC2=0 "$FULLPATH"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
