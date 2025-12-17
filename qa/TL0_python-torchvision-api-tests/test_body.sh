#!/bin/bash -e

test_py_with_framework() {
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A '!slow' -s torchvision
    else
        ${python_new_invoke_test} -A '!slow,!sanitizer_skip' -s torchvision
    fi
}

test_fw() {
    test_py_with_framework
}

run_all() {
  test_fw
}
