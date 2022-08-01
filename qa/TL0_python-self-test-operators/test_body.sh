#!/bin/bash -e

test_py_with_framework() {
    for test_script in $(ls operator/test_*.py); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
