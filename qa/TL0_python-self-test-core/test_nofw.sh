#!/bin/bash -e
# used pip packages
pip_packages="nose numpy>=1.17 opencv-python pillow nvidia-ml-py==11.450.51 numba"

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
