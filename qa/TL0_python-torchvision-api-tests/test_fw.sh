#!/bin/bash -e
# used pip packages
pipe_packagtes='${python_test_runner_package} numpy torch torchvision'

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
    test_fw
}

pushd ../..
source ./qa/test_template.sh
popd
