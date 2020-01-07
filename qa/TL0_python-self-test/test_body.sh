#!/bin/bash -e

test_nose() {
    for test_script in $(ls test_operator_*.py test_pipeline*.py test_backend_impl.py); do
        nosetests --verbose --attr '!slow' ${test_script}
    done
}

test_py() {
    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -i 10
    python test_coco_tfrecord.py -i 64
}

test_body() {
    test_nose
    test_py
}
