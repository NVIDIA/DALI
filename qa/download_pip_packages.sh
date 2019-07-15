#!/usr/bin/env bash

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}

pip_packages="nose matplotlib jupyter numpy torch tensorflow-gpu mxnet-cu${CUDA_VERSION} torchvision pillow opencv-python"
echo ${CUDA_VERSION}
mkdir /pip-packages
to_download=$(/opt/dali/qa/setup_packages.py -a -u ${pip_packages} --cuda ${CUDA_VERSION})
for p in ${to_download}; do
    pip download ${p} -d /pip-packages
done
