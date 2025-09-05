#!/bin/bash -e
# used pip packages
# don't gather deps for xavier test
if [ -z "$gather_pip_packages" ]
then
  # due to https://github.com/numpy/numpy/issues/18131 we cannot use 1.19.5
  pip_packages='${python_test_runner_package} dataclasses numpy>=1.23 opencv-python-headless pillow psutil astropy'
fi

target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
