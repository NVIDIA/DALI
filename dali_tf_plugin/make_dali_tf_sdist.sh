#!/bin/bash

set -o xtrace
set -e

PREBUILT_DIR=/prebuilt
echo "Listing available *.so files"
ls ${PREBUILT_DIR}/*/*.so

mv ${PREBUILT_DIR}/* .

mkdir -p dali_tf_sdist_build
cd dali_tf_sdist_build

cmake .. \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} \
      -DGIT_SHA=${GIT_SHA}

make -j
python setup.py sdist
cp dist/*.tar.gz /dali_tf_sdist
