#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy jax'
target_dir=./dali/test/python

one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    ${python_invoke_test} . test_jax_integration
}

pushd ../..
source ./qa/test_template.sh
popd
