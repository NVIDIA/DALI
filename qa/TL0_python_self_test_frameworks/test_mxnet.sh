#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy mxnet psutil'
target_dir=./dali/test/python

test_body() {
    ${python_invoke_test} -m '(?:^|[\b_\./-])[Tt]est.*mxnet' test_dltensor_operator.py
    ${python_invoke_test} test_external_source_parallel_mxnet.py
    ${python_invoke_test} --attr 'mxnet' test_external_source_impl_utils.py
    ${python_invoke_test} --attr 'mxnet' test_pipeline_debug.py
}

pushd ../..
source ./qa/test_template.sh
popd
