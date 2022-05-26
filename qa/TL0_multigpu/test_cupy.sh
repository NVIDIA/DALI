#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy cupy'
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_cupy
}

pushd ../..
source ./qa/test_template.sh
popd
