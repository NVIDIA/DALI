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
    FULLPATH=""
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

    # Invoke the test binary with an absolute path so
    # https://google.github.io/googletest/advanced.html#death-test-styles tests that run in
    # a separate process do not rely on PATH to find the executable.
    # PackedBFrames test is disabled because it doesn't work with the conda upstream build
    # of FFMpeg
    DALI_USE_EXEC2=0 "$FULLPATH" --gtest_filter="*:-*PackedBFrames*"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
