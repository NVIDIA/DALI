#!/bin/bash

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}
PYTHON_VERSION=$(python -c "from __future__ import print_function; import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")

NVIDIA_SMI_DRIVER_VERSION=$(nvidia-smi | grep -Po '(?<=Driver Version: )\d+.\d+')

function version_gt() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"; }
function version_le() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" == "$1"; }
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }
function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }

# If driver version is less than 410 and CUDA version is 10,
# add /usr/local/cuda/compat to LD_LIBRARY_PATH
version_ge "$CUDA_VERSION" "100" && \
version_lt "$NVIDIA_SMI_DRIVER_VERSION" "410.0" && \
export LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"