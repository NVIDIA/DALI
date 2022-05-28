#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy cupy'
target_dir=./dali/test/python

test_body() {
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*cupy' test_dltensor_operator.py
    ${python_invoke_test} test_gpu_python_function_operator.py
    ${python_invoke_test} test_backend_impl_gpu.py
    ${python_invoke_test} test_external_source_cupy.py
    ${python_invoke_test} --attr 'cupy' test_external_source_impl_utils.py
    ${python_invoke_test} --attr 'cupy' test_pipeline_debug.py
}

pushd ../..
source ./qa/test_template.sh
popd
