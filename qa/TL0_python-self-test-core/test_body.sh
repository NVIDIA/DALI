#!/bin/bash -e

test_py_with_framework() {
    # Note that we do not filter '!numba' below as it is installed as dependency
    for test_script in $(ls test_pipeline.py \
                            test_pipeline_debug.py \
                            test_pipeline_debug_resnet50.py \
                            test_eager_coverage.py \
                            test_eager_operators.py \
                            test_pipeline_decorator.py \
                            test_pipeline_multichannel.py \
                            test_pipeline_segmentation.py \
                            test_triton_autoserialize.py \
                            test_functional_api.py \
                            test_dali_variable_batch_size.py \
                            test_external_source_impl_utils.py); do
        if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
            ${python_invoke_test} --attr "!slow,!pytorch,!mxnet,!cupy" ${test_script}
        else
            ${python_invoke_test} --attr "!slow,!pytorch,!mxnet,!cupy,!numba" ${test_script}
        fi
    done

    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy' test_backend_impl

    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A 'numba' -s type_annotations
        ${python_new_invoke_test} -A '!slow,numba' checkpointing.test_dali_checkpointing
        ${python_new_invoke_test} -A '!slow,numba' checkpointing.test_dali_stateless_operators
    fi
}

test_py() {
    python test_python_function_cleanup.py
    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -s -i 10 --decoder_type "legacy"
    python test_RN50_data_pipeline.py -s -i 10 --decoder_type "experimental"
    python test_coco_tfrecord.py -i 64
    python test_data_containers.py -s -b 20
    python test_data_containers.py -s -b 20 -n
}

test_autograph() {
    ${python_new_invoke_test} -s autograph
    ${python_new_invoke_test} -s conditionals
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -s auto_aug
    fi
}

test_type_annotations() {
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A '!pytorch,!numba' -s type_annotations
    fi
}


test_experimental_mode_torch() {
    ${python_new_invoke_test}  -A 'pytorch' -s experimental_mode
}

test_pytorch() {
    ${python_invoke_test} --attr '!slow,pytorch' test_dali_variable_batch_size.py
    test_experimental_mode_torch
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A 'pytorch' -s type_annotations
        ${python_new_invoke_test} -A 'pytorch' -s dlpack
        ${python_new_invoke_test} -A '!slow' checkpointing.test_dali_checkpointing_fw_iterators.TestPytorch
        ${python_new_invoke_test} -A '!slow' checkpointing.test_dali_checkpointing_fw_iterators.TestPytorchRagged
    fi
}

test_checkpointing() {
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' checkpointing.test_dali_checkpointing
        ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' checkpointing.test_dali_stateless_operators
    else
        ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba,!sanitizer_skip' checkpointing.test_dali_checkpointing

        # External source tests are slow and Python-side mostly, but let's run just one of them
        ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' checkpointing.test_dali_checkpointing.test_external_source_checkpointing:1
    fi
}

test_experimental_mode() {
    ${python_new_invoke_test}  -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s experimental_mode
}


test_no_fw() {
    test_py_with_framework
    test_py
    test_autograph
    test_type_annotations
    test_checkpointing
    test_experimental_mode
}

run_all() {
  test_no_fw
  test_pytorch
}
