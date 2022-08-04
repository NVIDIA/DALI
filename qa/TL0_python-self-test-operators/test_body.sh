#!/bin/bash -e

test_py_with_framework() {
    for test_script in $(ls operator/test_*.py); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done

    ${python_new_invoke_test} -A '!slow' -s operator
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
