#!/usr/bin/env bash

pip_packages="nose matplotlib jupyter numpy torch tensorflow-gpu mxnet-cu##CUDA_VERSION## torchvision pillow"
pip_packages=$(echo ${pip_packages} | sed "s/##CUDA_VERSION##/${CUDA_VERSION}/")

mkdir /pip-packages
pip download $(/opt/dali/qa/setup_packages.py -a -u ${pip_packages} --cuda ${CUDA_VERSION}) -d /pip-packages



