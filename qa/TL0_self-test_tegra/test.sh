#!/bin/bash -ex

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    FULLPATH="$(find_test_bin "$BINNAME")"

    # LMDB seems to be greedy when mmaps memory, disable it as well
    # for some reason mmap based test tends to fail on some runners due to disc issue, so
    # disable it for now
    DALI_USE_EXEC2=0 "$FULLPATH" --gtest_filter=-*mmap*:*LMDBTest*
  done
}

pushd ../..
source ./qa/test_template.sh
popd
