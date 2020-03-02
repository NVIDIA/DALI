#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

export CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1\2/')

# Adding conda-forge channel for dependencies
conda config --add channels conda-forge

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

# Note: To building dependency packages:
# run 'install_requirements.txt' before installing DALI conda package)

export DALI_CONDA_BUILD_VERSION=$(cat ../VERSION)$(if [ "${NVIDIA_DALI_BUILD_FLAVOR}" != "" ]; then \
                                                     echo .${NVIDIA_DALI_BUILD_FLAVOR}.${DALI_TIMESTAMP}; \
                                                   fi)

# Building DALI package
conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/*.tar.bz2 artifacts
