#!/bin/bash -e
# used pip packages
pip_packages="tensorflow-gpu torchvision mxnet-cu##CUDA_VERSION##"
one_config_only=true

pushd ../..

cd dali/test/python

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    # TensorFlow doesn't support Python 3.7 yet
    if [ $PYTHON_VERSION != "3.7" ]; then
        python test_RN50_data_fw_iterators.py --gpus ${NUM_GPUS} -b 13 --workers 3 --prefetch 2 --epochs 3
    else
        python test_RN50_data_fw_iterators.py --gpus ${NUM_GPUS} -b 13 --workers 3 --prefetch 2 --epochs 3 --dissable_tf
    fi
}

source ../../../qa/test_template.sh

popd
