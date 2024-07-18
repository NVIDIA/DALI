#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa==0.10.2 torch psutil cupy'
target_dir=./dali/test/python

test_body() {
  ${python_new_invoke_test} test_pipeline_inputs
}

pushd ../..
source ./qa/test_template.sh
popd
