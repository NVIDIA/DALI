#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} tensorflow-gpu'

target_dir=./dali/test/python


test_body() {
  # skip TF tests for sanitizers as it leads to stack-overflow
  if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
    # CPU only test, remove CUDA from the search path just in case
    export LD_LIBRARY_PATH=""
    export PATH=${PATH/cuda/}
    ${python_invoke_test} test_dali_tf_plugin_cpu_only.py
    ${python_invoke_test} test_dali_tf_plugin_cpu_only_dataset.py
  fi
}

pushd ../..
source ./qa/test_template.sh
popd
