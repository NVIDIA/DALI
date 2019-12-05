#!/bin/bash -e
# used pip packages
pip_packages="nose numpy mxnet-cu{cuda_v}"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*mxnet' test_dltensor_operator.py
}

pushd ../..
source ./qa/test_template.sh
popd
