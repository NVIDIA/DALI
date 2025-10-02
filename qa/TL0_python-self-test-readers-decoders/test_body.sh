#!/bin/bash -e

test_py_with_framework() {
    # numpy seems to be extremly slow with sanitizers to dissable it
    if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
        FILTER_PATTERN="test_external_source_parallel.py\|test_external_source_parallel_custom_serialization\|test_external_source_parallel_garbage_collection_order"
    else
        FILTER_PATTERN="#"
    fi

    for test_script in $(ls test_external_source_dali.py test_external_source_numpy.py \
      test_external_source_parallel_garbage_collection_order.py \
      test_external_source_parallel_custom_serialization.py \
      test_pool.py test_external_source_parallel.py test_external_source_parallel_shared_batch.py \
      test_external_source_parallel_large_sample.py \
      | sed "/$FILTER_PATTERN/d"); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done


    if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
      SKIP_TESTS="test_numpy.py"
      READER_TESTS=""

      for test_script in $(ls reader/test_*); do
          test_script=(${test_script//// })
          test_script=${test_script[1]}

          if [[ "$SKIP_TESTS" != *"$test_script"* ]]; then
              test_script=${test_script::-3}
              READER_TESTS="$READER_TESTS $test_script"
          fi
      done

      ${python_new_invoke_test} -s reader $READER_TESTS
    else
      ${python_new_invoke_test} -s reader
    fi


    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
      ${python_new_invoke_test} -A '!jpeg_scans_limit' -s decoder
    else
      ${python_new_invoke_test} -A '!jpeg_scans_limit,!sanitizer_skip' -s decoder
    fi

    ${python_new_invoke_test} -s input
}

test_jpeg_scan_limit() {
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
      # test various broken cases with smaller limit to make the test faster
      DALI_MAX_JPEG_SCANS=30 ${python_new_invoke_test} -s decoder test_jpeg_scan_limit
      # test default limit for one case
      ${python_new_invoke_test} -s decoder test_jpeg_scan_limit.ProgressiveJpeg.test_scans_limit:1
    else
      # let's check if error handling does not lead to leaks
      DALI_MAX_JPEG_SCANS=30 ${python_new_invoke_test} -s decoder test_jpeg_scan_limit.ProgressiveJpeg.test_scans_limit:1
    fi
}

test_no_fw() {
    test_py_with_framework
    test_jpeg_scan_limit
}

run_all() {
  test_no_fw
}
