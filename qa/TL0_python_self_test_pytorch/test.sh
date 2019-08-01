#!/bin/bash -e
# used pip packages
pip_packages="nose numpy torch"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose test_pytorch_operator
}

pushd ../..
source ./qa/test_template.sh
popd
