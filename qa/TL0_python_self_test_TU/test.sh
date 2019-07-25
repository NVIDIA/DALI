#!/bin/bash -e
# used pip packages
pip_packages="nose numpy"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

test_body() {
    # workaround for the CI
    put_optflow_libs
    nosetests --verbose test_optical_flow.py
}

source ../../../qa/test_template.sh

popd
