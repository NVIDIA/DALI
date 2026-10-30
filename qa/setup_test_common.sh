#!/bin/bash

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')
CUDA_VERSION=${CUDA_VERSION:-100}
CUDA_VERSION_MAJOR=${CUDA_VERSION:0:2}

PYTHON_VERSION=$(python -c "import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")
PYTHON_VERSION_SHORT=${PYTHON_VERSION/\./}
if python3 -c "import sysconfig; exit(not sysconfig.get_config_var('Py_GIL_DISABLED'))"; then
  PYTHON_VERSION_SHORT="${PYTHON_VERSION_SHORT}t"
fi

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

# Exits with an error when any of the given paths does not exist. Use it for test data
# (e.g. datasets under /data) that has to be mounted in the test container, so the test
# fails fast instead of mid-run. It exits rather than returns, so it also works without `set -e`.
check_required_paths() {
    local path
    local -a missing=()
    for path in "$@"; do
        if [ ! -e "$path" ]; then
            missing+=("$path")
        fi
    done
    if [ ${#missing[@]} -ne 0 ]; then
        echo "ERROR: Test precondition failed. Required path(s) not found: ${missing[*]}." \
             "Make sure the required data is mounted in the container." >&2
        exit 1
    fi
}

COCO_DIR=/data/coco/coco-2017/coco2017

# Checks that COCO 2017 is mounted and exports DATA_DIR pointing to a directory with train2017,
# val2017 and annotations. When only the zip archives are present, they are extracted to a
# temporary directory and IS_TMP_DIR=1 is exported; the caller is responsible for removing it.
prepare_coco_dataset() {
    local archive
    local -a archives=(train2017.zip val2017.zip annotations_trainval2017.zip)
    check_required_paths "${COCO_DIR}"
    export DATA_DIR=${COCO_DIR}
    export IS_TMP_DIR=0
    if [ ! -f "${COCO_DIR}/train2017/000000581929.jpg" ] && [ -f "${COCO_DIR}/train2017.zip" ]; then
        command -v unzip >/dev/null || (apt update && apt install -y unzip)
        export DATA_DIR=$(mktemp -d)
        export IS_TMP_DIR=1
        pushd "${DATA_DIR}"
        for archive in "${archives[@]}"; do
            cp "${COCO_DIR}/${archive}" . &
        done
        wait
        for archive in "${archives[@]}"; do
            unzip -q "${archive}" &
        done
        wait
        popd
    fi
}

IMAGENET_DIR=/data/imagenet

# Checks that the ImageNet JPEG dataset is mounted and creates `train` and `val` links to it in
# the given directory (the current directory by default), unless they already exist.
link_imagenet_jpeg() {
    local dest=${1:-.}
    local split
    check_required_paths "${IMAGENET_DIR}/train-jpeg" "${IMAGENET_DIR}/val-jpeg"
    mkdir -p "${dest}"
    for split in train val; do
        if [ ! -d "${dest}/${split}" ]; then
            ln -sf "${IMAGENET_DIR}/${split}-jpeg/" "${dest}/${split}"
        fi
    done
}

# Finds a self-test binary and prints its canonical path. Pass `conda` to prefer
# the package installed in the active Conda environment; other suites prefer the
# local build tree so they test the code built by the current job.
find_test_bin() {
    local bin_name="$1"
    local prefer_conda_package="${2:-}"
    local package_dir
    local local_build_dir="../../build/dali/python/nvidia/dali"
    local candidate
    local -a candidates

    package_dir="$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' \
        2>/dev/null || echo '')"

    if [ "$prefer_conda_package" = "conda" ] && [ -n "${CONDA_PREFIX:-}" ] &&
       [ -n "$package_dir" ] &&
       [[ "$(readlink -f "$package_dir")" == "$(readlink -f "$CONDA_PREFIX")"/* ]]; then
        candidates=("$package_dir/test/$bin_name" "$local_build_dir/test/$bin_name")
    else
        candidates=("$local_build_dir/test/$bin_name" "$package_dir/test/$bin_name")
    fi

    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            readlink -f "$candidate"
            return 0
        fi
    done

    echo "ERROR: Self-test binary '$bin_name' was not found. Searched: ${candidates[*]}." >&2
    return 1
}

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
