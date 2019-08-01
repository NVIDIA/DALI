#!/bin/bash -e
# used pip packages
pip_packages="nose numpy"
target_dir=./dali/test/python

test_body() {
    nosetests --verbose test_plugin_manager.py
}

pushd ../..
source ./qa/test_template.sh
popd
