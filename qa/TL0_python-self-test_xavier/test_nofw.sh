#!/bin/bash -e
# used pip packages
# due to https://github.com/numpy/numpy/issues/18131 we cannot use 1.19.5
pip_packages="nose numpy>=1.17,<=1.19.4 opencv-python pillow psutil"

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
