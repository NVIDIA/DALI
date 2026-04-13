#!/bin/bash -xe

test_no_fw() {
    ${python_new_invoke_test} -A '!slow' free-threading experimental_mode.test_multithreading
}
