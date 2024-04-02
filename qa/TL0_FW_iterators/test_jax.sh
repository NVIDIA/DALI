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
    ${python_new_invoke_test} -A 'jax' test_fw_iterators

    # More specific JAX tests
    ${python_new_invoke_test} -s jax_plugin/ test_integration test_iterator test_peekable_iterator test_jax_operator
    ${python_new_invoke_test} checkpointing.test_dali_checkpointing_fw_iterators.TestJax
}

pushd ../..
source ./qa/test_template.sh
popd
