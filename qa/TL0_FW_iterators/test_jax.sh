#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy jax clu'
target_dir=./dali/test/python

one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    # General tests for iterators
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*jax*' test_fw_iterators.py

    # More specific JAX tests
    ${python_new_invoke_test} -s jax_plugin/ test_integration test_iterator test_peekable_iterator
}

pushd ../..
source ./qa/test_template.sh
popd
