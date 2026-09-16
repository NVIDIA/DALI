#!/bin/bash -ex

test_body() {
  for BINNAME in \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    FULLPATH="$(find_test_bin "$BINNAME")"

    DALI_USE_EXEC2=1 "$FULLPATH"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
