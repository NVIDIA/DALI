#!/bin/bash -ex

test_body() {
  for BINNAME in \
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

    # LMDB seems to be greedy when mmaps memory, disable it as well
    # for some reason mmap based test tends to fail on some runners due to disc issue, so
    # disable it for now
    DALI_USE_EXEC2=1 "$FULLPATH" --gtest_filter=-*mmap*:*LMDBTest*
  done
}

pushd ../..
source ./qa/test_template.sh
popd
