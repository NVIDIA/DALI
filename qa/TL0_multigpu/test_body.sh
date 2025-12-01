#!/bin/bash -e

test_py_with_framework() {
  # placeholder function
  :
}

test_py() {
  ${python_new_invoke_test} -s decoder -A 'multi_gpu'
  ${python_new_invoke_test} -A '!slow,!pytorch,!mxnet,!cupy,!numba,multi_gpu' -s experimental_mode
}

test_gtest() {
    for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    for DIRNAME in \
      "../../build/dali/python/nvidia/dali" \
      "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
    do
        if [ -x "$DIRNAME/test/$BINNAME" ]; then
            FULLPATH="$DIRNAME/test/$BINNAME"
            break
        fi
    done

    if [[ -z "$FULLPATH" ]]; then
        echo "ERROR: $BINNAME not found"
        exit 1
    fi

    "$FULLPATH" --gtest_filter="*MultiGPU*"
  done
}

test_cupy() {
    ${python_invoke_test} --attr 'multigpu' test_external_source_cupy.py
}


test_pytorch() {
    ${python_invoke_test} --attr 'multigpu' test_external_source_pytorch_gpu.py
    ${python_new_invoke_test} -A 'pytorch,multi_gpu' -s experimental_mode
}


test_jax() {
    # Workaround for NCCL version mismatch
    # TODO: Fix this in the CI setup_packages.py
    # or move this test to the L3 with JAX container as base
    echo "DALI_CUDA_VERSION_MAJOR=$DALI_CUDA_MAJOR_VERSION"
    if [ "$DALI_CUDA_MAJOR_VERSION" == "12" ]
    then
      python -m pip uninstall -y jax jaxlib
      python -m pip install --upgrade "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

      CUDA_VISIBLE_DEVICES="0,1" ${python_new_invoke_test} -s jax_plugin test_multigpu

      CUDA_VISIBLE_DEVICES="1" python jax_plugin/jax_client.py &
      CUDA_VISIBLE_DEVICES="0" python jax_plugin/jax_server.py

      CUDA_VISIBLE_DEVICES="1" python jax_plugin/jax_op_multi_gpu_sharding.py --id 1 &
      CUDA_VISIBLE_DEVICES="0" python jax_plugin/jax_op_multi_gpu_sharding.py --id 0
    fi
}

test_no_fw() {
    test_py_with_framework
    test_py
    test_gtest
}

run_all() {
  test_no_fw
  test_pytorch
}
