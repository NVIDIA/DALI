#!/bin/bash -ex

pip_packages='${python_test_runner_package} numpy'

target_dir=./dali/test/python

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin" \
    "dali_imgcodec_test.bin"
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
}

pushd ../..
source ./qa/test_template.sh
popd
