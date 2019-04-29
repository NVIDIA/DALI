#!/bin/bash -e
# used pip packages
pip_packages="nose numpy"

pushd ../..

source qa/setup_test.sh
cd dali/test/python

test_body() {
    nosetests --verbose test_plugin_manager.py
}

source ../../../qa/test_template.sh

popd
