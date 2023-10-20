#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy jax'

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_jax
}

# run this only on x86_64, not arm
if [ $(uname -m) != "x86_64" ]
then
  exit 0
fi

pushd ../..
source ./qa/test_template.sh
popd
