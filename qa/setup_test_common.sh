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

version_eq "$DALI_CUDA_MAJOR_VERSION" "13" && \
test "$NVIDIA_SMI_DRIVER_VERSION" != "" && \
version_lt "$NVIDIA_SMI_DRIVER_VERSION" "580.0" && \
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH"

echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"


# Solve the issues: ImportError:
# /usr/local/lib/python3.9/dist-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
# /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
# Seems there's an issue with libc:
# https://bugzilla.redhat.com/show_bug.cgi?id=1722181
# A fix has been proposed here:
# https://sourceware.org/ml/libc-alpha/2020-01/msg00099.html
preload_static_tls_libs() {
    if [ "$(uname -m)" = "aarch64" ] && [ -f /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0 ] ; then
        if [ -z "$LD_PRELOAD" ]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
        else
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD"
        fi
        export NEW_LD_PRELOAD=`find $(python -c "import sklearn; import os; print(os.path.dirname(sklearn.__file__))")/../scikit_learn.libs/ -name *libgomp-*.so.*`
        if [ -n "$NEW_LD_PRELOAD" ]; then
            export LD_PRELOAD="$NEW_LD_PRELOAD:$LD_PRELOAD"
        fi
    fi
}

preload_static_tls_libs


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

install_cuda_compat() {
    if [ "${DALI_CUDA_MAJOR_VERSION}" == "13" ] && [ "${CUDA_VERSION}" != "130" ]; then
        ARCH=$(uname -m)
        if [ "$ARCH" == "x86_64" ]; then
            REPO_ARCH="x86_64"
        elif [ "$ARCH" == "aarch64" ]; then
            REPO_ARCH="sbsa"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
        apt-get update && \
        apt-get install software-properties-common -y --no-install-recommends && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${REPO_ARCH}/3bf863cc.pub && \
        add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${REPO_ARCH}/ /" && \
        apt update && apt install -y cuda-compat-13-0
    fi
}
