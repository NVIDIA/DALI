#!/bin/bash -e
# used pip packages
pip_packages="nose numpy mxnet psutil"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*mxnet' test_dltensor_operator.py
    nosetests --verbose test_external_source_parallel_mxnet.py
    nosetests --verbose --attr 'mxnet' test_external_source_impl_utils.py
}

pushd ../..
source ./qa/test_template.sh
popd
