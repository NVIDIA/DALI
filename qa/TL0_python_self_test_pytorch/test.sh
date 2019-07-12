#!/bin/bash -e
# used pip packages
pip_packages="nose numpy torch"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

test_body() {
    nosetests --verbose test_pytorch_operator
}

source ../../../qa/test_template.sh

popd
