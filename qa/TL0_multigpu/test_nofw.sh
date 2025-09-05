#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa scipy nvidia-ml-py==11.450.51 psutil dill cloudpickle opencv-python-headless pillow'
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
