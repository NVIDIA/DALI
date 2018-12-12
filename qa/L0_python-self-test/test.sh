#!/bin/bash -e
# used pip packages
pip_packages="nose opencv-python numpy"

pushd ../..

cd dali/test/python

test_body() {
    # test code
    for test_script in $(ls *.py); do
        nosetests --verbose $test_script
    done
}

source ../../../qa/test_template.sh

popd
