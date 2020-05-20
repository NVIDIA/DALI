#!/bin/bash

set -o xtrace
set -e

DST_DIR=${1:-"/dali_tf_dummy"}

mkdir -p build
pushd build

CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1.\2/')

cmake .. \
      -DCUDA_VERSION:STRING="${CUDA_VERSION}" \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} \
      -DGIT_SHA=${GIT_SHA}

python setup.py sdist
cp dist/*.tar.gz ${DST_DIR}

popd
rm -rf build
