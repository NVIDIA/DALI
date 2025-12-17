#!/bin/bash -xe

test_no_fw() {
    ${python_new_invoke_test} -A '!slow' -s free-threading
}

