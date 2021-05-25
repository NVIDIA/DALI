#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python pillow nvidia-ml-py==11.450.51 torch numba"
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(: enable_conda)
epilog=(: disable_conda)

test_body() {
    nosetests --verbose test_optical_flow.py
    nosetests --verbose test_dali_variable_batch_size.py:test_optical_flow
}

pushd ../..
source ./qa/test_template.sh
popd
