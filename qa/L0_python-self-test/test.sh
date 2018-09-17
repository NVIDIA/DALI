#!/bin/bash -e
# used pip packages
pip_packages="nose opencv-python numpy"

pushd ../..

cd dali/test/python

test_body() {
    # test code
    nosetests --verbose test_pipeline.py
}

source ../../../qa/test_template.sh

popd
