#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy librosa==0.8.1 torch psutil'
target_dir=./dali/test/python

test_body() {
    ${python_test_runner} ${python_test_args} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator.py
    ${python_test_runner} ${python_test_args} -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator.py
    ${python_test_runner} ${python_test_args} test_torch_pipeline_rnnt.py
    ${python_test_runner} ${python_test_args} test_external_source_pytorch_cpu.py
    ${python_test_runner} ${python_test_args} test_external_source_pytorch_gpu.py
    ${python_test_runner} ${python_test_args} test_external_source_pytorch_dlpack.py
    ${python_test_runner} ${python_test_args} test_external_source_parallel_pytorch.py
    ${python_test_runner} ${python_test_args} test_backend_impl_torch_dlpack.py
    ${python_test_runner} ${python_test_args} test_dali_fork_torch.py
    ${python_test_runner} ${python_test_args} --attr 'pytorch' test_external_source_impl_utils.py
    ${python_test_runner} ${python_test_args} --attr 'pytorch' test_pipeline_debug.py
    ${python_test_runner} ${python_test_args} --attr 'pytorch' test_functional_api.py
}

pushd ../..
source ./qa/test_template.sh
popd
