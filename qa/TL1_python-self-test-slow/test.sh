#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python-headless pillow librosa scipy nvidia-ml-py==11.450.51 numba lz4'
target_dir=./dali/test/python

test_body() {
    for test_script in $(ls test_pipeline.py test_pipeline_debug.py test_pipeline_debug_resnet50.py \
                            test_pipeline_decorator.py test_pipeline_multichannel.py test_pipeline_segmentation.py \
                            test_functional_api.py); do
        ${python_invoke_test} --attr 'slow' ${test_script}
    done

    ${python_new_invoke_test} -A "slow" test_backend_impl

    ${python_new_invoke_test} --config unittest_slow.cfg -s operator_1
    ${python_new_invoke_test} --config unittest_slow.cfg -s operator_2
}

pushd ../..
source ./qa/test_template.sh
popd
