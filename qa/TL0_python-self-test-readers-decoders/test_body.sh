#!/bin/bash -e

test_py_with_framework() {
    # numpy seems to be extremly slow with sanitizers to dissable it
    if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
          FILTER_PATTERN="test_operator_readers_numpy.py\|test_external_source_parallel.py\|test_external_source_parallel_custom_serialization\|test_external_source_parallel_garbage_collection_order"
    else
        FILTER_PATTERN="#"
    fi

    for test_script in $(ls test_operator_readers_*.py test_operator_decoders_*.py \
      test_external_source_dali.py test_external_source_numpy.py \
      test_external_source_parallel_garbage_collection_order.py \
      test_external_source_parallel_custom_serialization.py \
      test_pool.py test_external_source_parallel.py test_external_source_parallel_shared_batch.py \
      test_external_source_parallel_large_sample.py \
      | sed "/$FILTER_PATTERN/d"); do
        ${python_invoke_test} --attr '!slow,!pytorch,!mxnet,!cupy,!numba' ${test_script}
    done
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
