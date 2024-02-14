#!/bin/bash -e

test_py_with_framework() {
    for test_script in $(ls test_pipeline*.py test_external_source_dali.py test_external_source_numpy.py test_external_source_parallel_garbage_collection_order.py test_functional_api.py test_backend_impl.py); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done

    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s operator_1
    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s operator_2
    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s reader
    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s decoder
}

test_py() {
    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -s -i 10 --decoder_type "legacy"
    python test_RN50_data_pipeline.py -s -i 10 --decoder_type "experimental"
    python test_coco_tfrecord.py -i 64
    python test_data_containers.py -s -b 20
    python test_data_containers.py -s -b 20 -n
}

test_no_fw() {
    test_py_with_framework
    test_py
}

run_all() {
  test_no_fw
}
