#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa torch psutil cupy'
target_dir=./dali/test/python

test_body() {
  ${python_new_invoke_test} test_pipeline_inputs
}

pushd ../..
source ./qa/test_template.sh
popd
