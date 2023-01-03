#!/bin/bash -e

test_py_with_framework() {
    ${python_new_invoke_test} -A '!slow' -s operator_2
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
