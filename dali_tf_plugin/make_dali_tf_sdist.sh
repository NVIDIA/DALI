#!/bin/bash

set -o xtrace
set -e

PREBUILT_DIR="./prebuilt"
echo "Listing available *.so files:"
find . -name '*.so' || true

mkdir -p dali_tf_sdist_build
pushd dali_tf_sdist_build

CUDA_VERSION_STR=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2.\3/')

cmake .. \
      -DCUDA_VERSION:STRING="${CUDA_VERSION_STR}" \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} \
      -DGIT_SHA=${GIT_SHA}
make -j install
python setup.py sdist
mkdir -p /dali_tf_sdist
cp dist/*.tar.gz /dali_tf_sdist
popd
