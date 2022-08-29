#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python pillow librosa==0.8.1 scipy nvidia-ml-py==11.450.51 numba'
target_dir=./dali/test/python

test_body() {
    for test_script in $(ls test_pipeline*.py test_functional_api.py test_backend_impl.py); do
        ${python_invoke_test} --attr 'slow' ${test_script}
    done

    ${python_new_invoke_test} --config unittest_slow.cfg -s operator
}

pushd ../..
source ./qa/test_template.sh
popd
