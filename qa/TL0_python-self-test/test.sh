#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python pillow librosa"
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

pushd ../..
source ./qa/test_template.sh
popd
