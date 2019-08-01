#!/bin/bash -e
# used pip packages
pip_packages="nose numpy"
target_dir=./dali/test/python

test_body() {
    # workaround for the CI
    put_optflow_libs
    nosetests --verbose test_optical_flow.py
}


pushd ../..
source ./qa/test_template.sh
popd
