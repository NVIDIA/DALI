#!/bin/bash -e

test_py_with_framework() {
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A '!slow' -s operator_2
    else
        ${python_new_invoke_test} -A '!slow,!sanitizer_skip' -s operator_2
    fi
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
