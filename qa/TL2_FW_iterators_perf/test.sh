#!/bin/bash -e
# used pip packages
# TODO(janton): remove explicit pillow version installation when torch fixes the issue with PILLOW_VERSION not being defined
# rarfile>= 3.2 breaks python 3.5 compatibility
pip_packages="pillow==6.2.2 tensorflow-gpu torchvision mxnet torch paddle rarfile<=3.1"
target_dir=./dali/test/python
one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    for fw in "mxnet" "pytorch" "tf" "paddle"; do
        python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
            --workers 3 --prefetch 2 --epochs 3
    done
}

pushd ../..
source ./qa/test_template.sh
popd
