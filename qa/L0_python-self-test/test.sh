#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python pillow"

# test_body definition is in separate file so it can be used without setup
source test_body.sh

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

source ../../../qa/test_template.sh

popd
