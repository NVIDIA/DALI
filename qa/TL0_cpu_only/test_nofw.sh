#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy>=1.17 numba scipy librosa==0.8.1'

target_dir=./dali/test/python

test_body() {
  # CPU only test, remove CUDA from the search path just in case
  export LD_LIBRARY_PATH=""
  export PATH=${PATH/cuda/}

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

    "$FULLPATH" --gtest_filter="*CpuOnly*:*CApi*/0.*-*0.UseCopyKernel:*ForceNoCopyFail:*daliOutputCopySamples"
  done

  ${python_invoke_test} --attr '!pytorch' test_dali_cpu_only.py
}

pushd ../..
source ./qa/test_template.sh
popd
