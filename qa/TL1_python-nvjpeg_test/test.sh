#!/bin/bash -e
# used pip packages
pip_packages="numpy"
target_dir=./dali/test/python

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    # test code
    python test_data_containers.py --gpus ${NUM_GPUS} -b 2048 -p 10
}

pushd ../..
source ./qa/test_template.sh
popd
