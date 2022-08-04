#!/bin/bash -e

test_py_with_framework() {
    # Note that we do not filter '!numba' below as it is installed as dependency
    for test_script in $(ls test_pipeline*.py \
                            test_triton_autoserialize.py \
                            test_functional_api.py \
                            test_backend_impl.py \
                            test_dali_variable_batch_size.py \
                            test_external_source_impl_utils.py); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy' ${test_script}
    done
}

test_py() {
    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -s -i 10
    python test_coco_tfrecord.py -i 64
    python test_data_containers.py -s -b 20
    python test_data_containers.py -s -b 20 -n
}

test_pytorch() {
    ${python_invoke_test} --attr '!slow,pytorch' test_dali_variable_batch_size.py
}

test_no_fw() {
    test_py_with_framework
    test_py
}

run_all() {
  test_no_fw
  test_pytorch
}
