#!/bin/bash -e
# used pip packages
pip_packages=""

pushd ../..

cd dali/test/python

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    # test code
    python test_data_containers.py --gpus ${NUM_GPUS} -b 2048 -p 10
}

source ../../../qa/test_template.sh

popd
