#!/bin/bash -e

# Applies only to Python 3.XXt
export PYTHON_GIL=0

test_py_with_framework() {
    ${python_new_invoke_test} -A '!slow' -s free-threading
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}