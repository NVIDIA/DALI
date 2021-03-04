#!/bin/bash -e

test_nose() {
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
    )

    for test_script in $(ls test_operator_*.py test_pipeline*.py test_pool.py test_external_source_dali.py test_external_source_numpy.py test_external_source_parallel.py test_external_source_parallel_shared_batch.py test_functional_api.py test_backend_impl.py); do
        status=0
        for exclude in "${EXCLUDE_PACKAGES[@]}"; do
            grep -qiE ${exclude} ${test_script} && status=$((status+1))
        done
        # execute only when no matches are found
        if [ ${status} -eq 0 ]; then
            nosetests --verbose --attr '!slow' ${test_script}
        fi
    done
}

test_py() {
    :
}

test_no_fw() {
    test_nose
    test_py
}

run_all() {
  test_no_fw
}
