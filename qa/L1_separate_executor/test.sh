#!/bin/bash -e
# used pip packages
pip_packages="tensorflow-gpu torchvision mxnet-cu##CUDA_VERSION##"
one_config_only=true

pushd ../..

cd dali/test/python

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --separate_queue --cpu_size 2 --gpu_size 2 --fp16 --nhwc

    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --separate_queue --cpu_size 5 --gpu_size 3 --fp16 --nhwc

    python test_RN50_data_fw_iterators.py --gpus ${NUM_GPUS} -b 13 --workers 3 --separate_queue --cpu_size 3 --gpu_size 2 --iters 32 --epochs 2
}

source ../../../qa/test_template.sh

popd
