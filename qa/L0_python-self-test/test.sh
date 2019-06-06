#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python torchvision"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

test_body() {
    nosetests --verbose test_optical_flow.py
    nosetests --verbose test_backend_impl.py
    nosetests --verbose test_pipeline.py
    nosetests --verbose test_decoders.py
    nosetests --verbose test_python_function_operator.py

    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -i 10
}

source ../../../qa/test_template.sh

popd
