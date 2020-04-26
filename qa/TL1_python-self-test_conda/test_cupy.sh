#!/bin/bash -e
# used pip packages
pip_packages="nose numpy cupy"
target_dir=./dali/test/python

# test_body definition is in separate file so it can be used without setup
source test_body.sh

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
  test_cupy
}

pushd ../..
source ./qa/test_template.sh
popd
