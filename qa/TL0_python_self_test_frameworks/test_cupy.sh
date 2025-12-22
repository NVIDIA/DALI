#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy cupy pycuda'
target_dir=./dali/test/python

test_body() {
    ${python_new_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*cupy' test_dltensor_operator
    ${python_new_invoke_test} test_gpu_python_function_operator
    ${python_new_invoke_test} test_backend_impl_gpu
    ${python_new_invoke_test} test_external_source_cupy
    ${python_new_invoke_test} -A 'cupy' test_external_source_impl_utils
    ${python_new_invoke_test} -A 'cupy' test_pipeline_debug
    ${python_new_invoke_test} -A '!slow,cupy' checkpointing.test_dali_checkpointing
    ${python_new_invoke_test} -A '!slow,cupy' checkpointing.test_dali_stateless_operators
}

pushd ../..
source ./qa/test_template.sh
popd
