#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa torch psutil torchvision'
target_dir=./dali/test/python

test_body() {
    ${python_new_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator
    ${python_new_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator
    ${python_new_invoke_test} test_torch_pipeline_rnnt
    ${python_new_invoke_test} test_external_source_pytorch_cpu
    ${python_new_invoke_test} test_external_source_pytorch_gpu
    ${python_new_invoke_test} test_external_source_pytorch_dlpack
    ${python_new_invoke_test} test_external_source_parallel_pytorch
    ${python_new_invoke_test} test_backend_impl_torch_dlpack
    ${python_new_invoke_test} test_dali_fork_torch
    ${python_new_invoke_test} test_copy_to_external_torch
    ${python_new_invoke_test} -A 'pytorch' test_external_source_impl_utils
    ${python_new_invoke_test} -A 'pytorch' test_pipeline_debug
    ${python_new_invoke_test} -A 'pytorch' test_functional_api
    ${python_new_invoke_test} -s . test_dali_proxy
}

pushd ../..
source ./qa/test_template.sh
popd
