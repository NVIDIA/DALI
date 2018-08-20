#!/bin/bash -e
# used pip packages
pip_packages=""

pushd ../..

cd dali/test/python

test_body() {
    # test code
    python test_data_containers.py
}

source ../../../qa/test_template.sh

popd
