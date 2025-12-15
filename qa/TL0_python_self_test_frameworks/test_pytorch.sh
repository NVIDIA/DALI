#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa torch psutil torchvision'
target_dir=./dali/test/python

test_body() {
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator.py
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator.py
    ${python_invoke_test} test_torch_pipeline_rnnt.py
    ${python_invoke_test} test_external_source_pytorch_cpu.py
    ${python_invoke_test} test_external_source_pytorch_gpu.py
    ${python_invoke_test} test_external_source_pytorch_dlpack.py
    ${python_invoke_test} test_external_source_parallel_pytorch.py
    ${python_invoke_test} test_backend_impl_torch_dlpack.py
    ${python_invoke_test} test_dali_fork_torch.py
    ${python_invoke_test} test_copy_to_external_torch.py
    ${python_invoke_test} --attr 'pytorch' test_external_source_impl_utils.py
    ${python_invoke_test} --attr 'pytorch' test_pipeline_debug.py
    ${python_invoke_test} --attr 'pytorch' test_functional_api.py
    ${python_new_invoke_test} -s . test_dali_proxy
}

pushd ../..
source ./qa/test_template.sh
popd
