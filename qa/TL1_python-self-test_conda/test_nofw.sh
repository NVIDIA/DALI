#!/bin/bash -e
# used pip packages
# lock numba version as 0.50 changed module location and librosa hasn't catched up in 7.2 yet
pip_packages="nose numpy>=1.17 opencv-python pillow librosa numba<=0.49"
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
