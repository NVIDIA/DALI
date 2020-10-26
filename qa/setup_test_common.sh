#!/bin/bash

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')
CUDA_VERSION=${CUDA_VERSION:-100}

if [ -n "$gather_pip_packages" ]
then
    # early exit
    return 0
fi
PYTHON_VERSION=$(python -c "import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")
PYTHON_VERSION_SHORT=${PYTHON_VERSION/\./}

NVIDIA_SMI_DRIVER_VERSION=$(nvidia-smi | grep -Po '(?<=Driver Version: )\d+.\d+') || true

function version_gt() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"; }
function version_le() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" == "$1"; }
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }
function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }

# If driver version is less than 450 and CUDA version is 11,
# add /usr/local/cuda/compat to LD_LIBRARY_PATH
version_ge "$CUDA_VERSION" "110" && \
version_lt "$NVIDIA_SMI_DRIVER_VERSION" "450.0" && \
export LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"

enable_conda() {
    echo "Activate conda"
    # functions are not exported by default to be made available in subshells
    eval "$(conda shell.bash hook)"
    conda activate conda_py${PYTHON_VERSION_SHORT}_env
}

disable_conda() {
    echo "Deactivate conda"
    conda deactivate
}

enable_virtualenv() {
    echo "Activate virtual env"
    source /virtualenv_${PYTHON_VERSION_SHORT}/bin/activate
}

disable_virtualenv() {
    echo "Deactivate virtual env"
    deactivate
}
