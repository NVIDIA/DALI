#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} tensorflow-gpu'

target_dir=./dali/test/python


test_body() {
  # skip TF tests for sanitizers as it leads to stack-overflow
  if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
    # CPU only test, remove CUDA from the search path just in case.
    # Keep the nvimgcodec wheel directory so libnvimgcodec.so (CPU codecs via
    # libjpeg-turbo / libtiff / opencv) remains discoverable for dlopen — it
    # doesn't depend on system CUDA libs.
    export LD_LIBRARY_PATH="$(python -c 'import nvidia.nvimgcodec as n, os; print(os.path.dirname(n.__file__))' 2>/dev/null || echo '')"
    export PATH=${PATH/cuda/}
    ${python_new_invoke_test} test_dali_tf_plugin_cpu_only
    ${python_new_invoke_test} test_dali_tf_plugin_cpu_only_dataset
  fi
}

pushd ../..
source ./qa/test_template.sh
popd
