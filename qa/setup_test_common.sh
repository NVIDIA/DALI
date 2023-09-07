#!/bin/bash

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')
CUDA_VERSION=${CUDA_VERSION:-100}
CUDA_VERSION_MAJOR=${CUDA_VERSION:0:2}

PYTHON_VERSION=$(python -c "import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")
PYTHON_VERSION_SHORT=${PYTHON_VERSION/\./}

NVIDIA_SMI_DRIVER_VERSION=$(nvidia-smi | grep -Po '(?<=Driver Version: )\d+.\d+') || true

DALI_CUDA_MAJOR_VERSION=$(pip list | grep nvidia-dali.*-cuda | cut -d " " -f1) && \
                        DALI_CUDA_MAJOR_VERSION=${DALI_CUDA_MAJOR_VERSION#*cuda} && \
                        DALI_CUDA_MAJOR_VERSION=${DALI_CUDA_MAJOR_VERSION:0:2}

# if DALI is not present in the system just take CUDA_VERSION_MAJOR as we may just test DALI
# compilation process
test -z "${DALI_CUDA_MAJOR_VERSION}" && export DALI_CUDA_MAJOR_VERSION=${CUDA_VERSION_MAJOR}

function version_gt() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" != "$1"; }
function version_le() { test "$(echo "$@" | tr " " "\n" | sort -V | head -n 1)" == "$1"; }
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }
function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }
function version_eq() { test "$1" == "$2"; }

if [ -n "$gather_pip_packages" ]
then
    # early exit
    return 0
fi

# If driver version is less than 450 and CUDA version is 11,
# add /usr/local/cuda/compat to LD_LIBRARY_PATH
version_eq "$DALI_CUDA_MAJOR_VERSION" "11" && \
test "$NVIDIA_SMI_DRIVER_VERSION" != "" && \
version_lt "$NVIDIA_SMI_DRIVER_VERSION" "450.0" && \
export LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH"

version_eq "$DALI_CUDA_MAJOR_VERSION" "12" && \
test "$NVIDIA_SMI_DRIVER_VERSION" != "" && \
version_lt "$NVIDIA_SMI_DRIVER_VERSION" "525.0" && \
export LD_LIBRARY_PATH="/usr/local/cuda-12.0/compat:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"

enable_conda() {
    echo "Activate conda"
    # functions are not exported by default to be made available in subshells
    eval "$(conda shell.bash hook)"
    conda activate conda_py${PYTHON_VERSION_SHORT}_env
    # according to https://www.tensorflow.org/install/pip we need to make sure that
    # TF will use conda lib, not system one to link. Otherwise it will use the system libstdc++.so.6
    # and everything what is imported after it will use it as well
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
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
