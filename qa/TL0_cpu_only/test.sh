#!/bin/bash -e
# used pip packages
pip_packages="nose numpy>=1.17 pillow torch"

target_dir=./dali/test/python

test_body() {
  # CPU only test, remove CUDA from the search path just in case
  export LD_LIBRARY_PATH=""
  export PATH=${PATH/cuda/}
  nosetests --verbose test_dali_cpu_only.py
}

pushd ../..
source ./qa/test_template.sh
popd
