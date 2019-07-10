#!/usr/bin/env bash

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}

pip_packages="nose matplotlib jupyter numpy torch tensorflow-gpu mxnet-cu${CUDA_VERSION} torchvision pillow"
echo ${CUDA_VERSION}
mkdir /pip-packages
pip download $(/opt/dali/qa/setup_packages.py -a -u ${pip_packages} --cuda ${CUDA_VERSION}) -d /pip-packages



