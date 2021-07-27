#!/bin/bash -e
# used pip packages
pip_packages="nose numpy cupy"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*cupy' test_dltensor_operator.py
    nosetests --verbose test_gpu_python_function_operator.py
    nosetests --verbose test_backend_impl_gpu.py
    nosetests --verbose test_external_source_cupy.py
    nosetests --verbose --attr 'cupy' test_external_source_impl_utils.py
}

pushd ../..
source ./qa/test_template.sh
popd
