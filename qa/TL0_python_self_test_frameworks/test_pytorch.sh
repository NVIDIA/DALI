#!/bin/bash -e
# used pip packages
pip_packages="nose numpy torch"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_pytorch_operator.py
    nosetests --verbose -m '(?:^|[\b_\./-])[Tt]est.*pytorch' test_dltensor_operator.py
}

pushd ../..
source ./qa/test_template.sh
popd
