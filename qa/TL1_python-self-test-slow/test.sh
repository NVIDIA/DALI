#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python pillow librosa scipy nvidia-ml-py==11.450.51 numba"
target_dir=./dali/test/python

test_body() {
    for test_script in $(ls test_operator_*.py test_pipeline*.py test_functional_api.py test_backend_impl.py); do
        nosetests --verbose --attr 'slow' ${test_script}
    done
}

pushd ../..
source ./qa/test_template.sh
popd
