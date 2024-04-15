#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy jax clu'
target_dir=./dali/test/python

one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    # Updating JAX 0.4.13 -> 0.4.16 (or the most recent 0.4.26) causes
    # asan to fail with suposed stack-overflow
    # TODO(ktokarski) Investigate what may be the underlying cause
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        # General tests for iterators
        ${python_new_invoke_test} -A 'jax' test_fw_iterators
    fi

    # More specific JAX tests
    ${python_new_invoke_test} -s jax_plugin/ test_integration test_iterator test_peekable_iterator

    # Updating JAX 0.4.13 -> 0.4.16 (or the most recent 0.4.26) causes
    # asan to fail with suposed stack-overflow
    # TODO(ktokarski) Investigate what may be the underlying cause
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} checkpointing.test_dali_checkpointing_fw_iterators.TestJax
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
