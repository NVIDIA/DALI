#!/bin/bash -e

echo "test_body.sh Sanity check: PYTHON_GIL=${PYTHON_GIL}"

test_py_with_framework() {
    ${python_new_invoke_test} -A '!slow' -s free-threading
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
