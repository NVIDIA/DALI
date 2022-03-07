#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"

target_dir=./dali/test/python

test_body() {
  # CPU only test, remove CUDA from the search path just in case
  export LD_LIBRARY_PATH=""
  export PATH=${PATH/cuda/}
  nosetests --verbose test_dali_tf_plugin_cpu_only.py
  nosetests --verbose test_dali_tf_plugin_cpu_only_dataset.py
}

pushd ../..
source ./qa/test_template.sh
popd
