#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy cupy'
target_dir=./dali/test/python

test_body() {
    ${python_test_runner} ${python_test_args} -m '(?:^|[\b_\./-])[Tt]est.*cupy' test_dltensor_operator.py
    ${python_test_runner} ${python_test_args} test_gpu_python_function_operator.py
    ${python_test_runner} ${python_test_args} test_backend_impl_gpu.py
    ${python_test_runner} ${python_test_args} test_external_source_cupy.py
    ${python_test_runner} ${python_test_args} --attr 'cupy' test_external_source_impl_utils.py
    ${python_test_runner} ${python_test_args} --attr 'cupy' test_pipeline_debug.py
}

pushd ../..
source ./qa/test_template.sh
popd
