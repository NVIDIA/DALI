#!/bin/bash -e
# used pip packages
pip_packages="nose torch numba"

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  # run this only on x86_64, not arm
  if [ $(uname -m) == "x86_64" ]
  then
    test_pytorch
  fi
}

pushd ../..
source ./qa/test_template.sh
popd
