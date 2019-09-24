#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

nvidia-smi

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

# Building dependency packages
conda build ${CONDA_BUILD_OPTIONS} third_party/jpeg_turbo/recipe

# Building DALI package
conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/*.tar.bz2 artifacts
