#!/bin/bash -e

test_py_with_framework() {
    # we are not able to easily install this packages in xavier for aarch64 so filter it out
    # also there is no nvJPEG on xavier so don't run any test with the ImageDecoder having
    # the device explicitly set
    EXCLUDE_PACKAGES=(
        "scipy"
        "librosa"
        "\"mixed\""
        "'mixed'"
        "video\("
        "caffe"
        "numba"
    )

    for test_script in $(ls test_pipeline*.py test_pool.py test_external_source_dali.py test_external_source_numpy.py test_external_source_parallel.py test_external_source_parallel_shared_batch.py test_functional_api.py test_backend_impl.py); do
        status=0
        for exclude in "${EXCLUDE_PACKAGES[@]}"; do
            grep -qiE ${exclude} ${test_script} && status=$((status+1))
        done
        # execute only when no matches are found
        if [ ${status} -eq 0 ]; then
            ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
        fi
    done


    XAVIER_OPERATOR_TESTS=""
    for test_script in $(ls operator/test_*.py); do
        status=0
        for exclude in "${EXCLUDE_PACKAGES[@]}"; do
            grep -qiE ${exclude} ${test_script} && status=$((status+1))
        done
        # execute only when no matches are found
        if [ ${status} -eq 0 ]; then
            test_script=(${test_script//// })
            test_script=${test_script[1]::-3}
            XAVIER_OPERATOR_TESTS="$XAVIER_OPERATOR_TESTS $test_script"
        fi
    done

    XAVIER_READER_TESTS=""
    for test_script in $(ls reader/test_*.py); do
        status=0
        for exclude in "${EXCLUDE_PACKAGES[@]}"; do
            grep -qiE ${exclude} ${test_script} && status=$((status+1))
        done
        # execute only when no matches are found
        if [ ${status} -eq 0 ]; then
            test_script=(${test_script//// })
            test_script=${test_script[1]::-3}
            XAVIER_READER_TESTS="$XAVIER_READER_TESTS $test_script"
        fi
    done

    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s operator $XAVIER_OPERATOR_TESTS
    ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba' -s reader $XAVIER_READER_TESTS
}

test_py() {
    :
}

test_no_fw() {
    test_py_with_framework
    test_py
}

run_all() {
  test_no_fw
}
