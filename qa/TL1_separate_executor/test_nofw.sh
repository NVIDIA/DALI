#!/bin/bash -e
# used pip packages
pip_packages=""
target_dir=./dali/test/python
one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --separate_queue \
        --cpu_size 2 --gpu_size 2 --fp16 --nhwc
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --separate_queue \
        --cpu_size 5 --gpu_size 3 --fp16 --nhwc
}

pushd ../..
source ./qa/test_template.sh
popd
