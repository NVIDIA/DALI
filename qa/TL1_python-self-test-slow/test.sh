#!/bin/bash -e
# used pip packages
pip_packages='nose dataclasses numpy opencv-python pillow librosa==0.8.1 scipy nvidia-ml-py==11.450.51 numba'
target_dir=./dali/test/python

test_body() {
    for test_script in $(ls test_operator_*.py test_pipeline*.py test_functional_api.py test_backend_impl.py); do
        ${python_test_runner} ${python_test_args} --attr 'slow' ${test_script}
    done
}

pushd ../..
source ./qa/test_template.sh
popd
