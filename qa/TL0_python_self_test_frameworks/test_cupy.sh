#!/bin/bash -e
# used pip packages
pip_packages="nose numpy cupy"
target_dir=./dali/test/python

test_body() {
    if [[ ${PYTHON_VERSION} != 2.7 ]]
    then
        nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*cupy' test_dltensor_operator.py
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
