#!/bin/bash -e
# used pip packages
pip_packages="nose numpy torch mxnet-cu{cuda_v} cupy"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose test_pytorch_operator
    nosetests --verbose test_dltensor_operator
    nosetests --verbose test_async_outputs
}

pushd ../..
source ./qa/test_template.sh
popd
