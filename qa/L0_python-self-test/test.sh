#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python pillow"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

test_body() {
    nosetests --verbose test_backend_impl.py
    nosetests --verbose test_pipeline.py
    for test_script in $(ls test_operator_*.py); do
        nosetests --verbose ${test_script}
    done

    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -i 10
}

source ../../../qa/test_template.sh

popd
