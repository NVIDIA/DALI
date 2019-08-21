#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi
CONDA_BUILD_OPTIONS="--python=${PYVER}"

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1/')

if [ "$CUDA_VERSION" = "9" ]; then
    echo "CUDA 9 build"
elif [ "$CUDA_VERSION" = "10" ]; then
    echo "CUDA 10 build"
else
    echo "CUDA ${CUDA_VERSION} not supported (only CUDA 9 and 10 are supported)"
    exit 1
fi

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config_cuda_${CUDA_VERSION}.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

# Building dependency packages
conda build ${CONDA_BUILD_OPTIONS} third_party/jpeg_turbo/recipe

# Building DALI package
conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/*.tar.bz2 artifacts
