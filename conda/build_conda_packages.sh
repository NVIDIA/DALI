#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

nvidia-smi

# Adding conda-forge channel for dependencies
conda config --add channels conda-forge

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

# Note: To building dependency packages:
# run 'install_requirements.txt' before installing DALI conda package)

# Building DALI package
conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/*.tar.bz2 artifacts
