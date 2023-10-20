#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package}'
target_dir=./dali/test/python

test_body() {
    ${python_new_invoke_test} --config unittest_failure.cfg test_trigger_failure
}

pushd ../..
source ./qa/test_template.sh
popd
