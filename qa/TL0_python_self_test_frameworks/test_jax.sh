#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy jax clu'
target_dir=./dali/test/python

test_body() {
  ${python_new_invoke_test} -s jax_plugin/ test_jax_operator
}

pushd ../..
source ./qa/test_template.sh
popd
