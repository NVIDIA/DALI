#!/bin/bash -e
# used pip packages
pip_packages="nose numpy>=1.17 opencv-python pillow librosa"
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
  test_no_fw
}

pushd ../..
source ./qa/test_template.sh
popd
