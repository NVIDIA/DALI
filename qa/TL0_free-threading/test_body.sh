#!/bin/bash -xe

echo "test_body.sh Sanity check: PYTHON_GIL=${PYTHON_GIL}"

test_no_fw() {
    ${python_new_invoke_test} -A '!slow' -s free-threading
}

