#!/bin/bash -e

test_py_with_framework() {
  # placeholder function
  :
}

test_py() {
  # placeholder function
  :
}

test_gtest() {
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

    "$FULLPATH" --gtest_filter="*MultiGPU*"
  done
}

test_cupy() {
    ${python_invoke_test} --attr 'multigpu' test_external_source_cupy.py
}

test_pytorch() {
    ${python_invoke_test} --attr 'multigpu' test_external_source_pytorch_gpu.py
}

test_no_fw() {
    test_py_with_framework
    test_py
    test_gtest
}

run_all() {
  test_no_fw
  test_pytorch
}
