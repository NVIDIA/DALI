#!/bin/bash -e
# used pip packages

pip_packages="torchvision mxnet-cu##CUDA_VERSION##"

pushd ../..

cd dali/test/python

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    # We install it manually here because we want to limit to one version of TF
    # depending on the CUDA version. (note: CUDA 10 is only supported in versions > 1.13)
    if [[ "${CUDA_VERSION}" == "90" ]]; then
        pip install tensorflow-gpu==1.12.0
    elif [[ "${CUDA_VERSION}" == "100" ]]; then
        pip install tensorflow-gpu==1.13.1
    else
        echo "Not supported CUDA version"
        exit 1
    fi
    python test_RN50_data_fw_iterators.py --gpus ${NUM_GPUS} -b 13 --workers 3 --prefetch 2 -i 100 --epochs 2
}

source ../../../qa/test_template.sh

popd
