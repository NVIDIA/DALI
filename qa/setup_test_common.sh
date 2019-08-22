#!/bin/bash

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}

if [ -n "$gather_pip_packages" ]
then
    # early exit
    return 0
fi
PYTHON_VERSION=$(python -c "from __future__ import print_function; import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")
PYTHON_VERSION_SHORT=${PYTHON_VERSION/\./}

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

put_optflow_libs() {
  # Docker v1 doesn't expose libnvidia-opticalflow.so from the host so we need to manually put it there
  # workaround for the CI
  if [[ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so ]]; then
      NVIDIA_SMI_DRIVER_VERSION_LONG=$(nvidia-smi | grep -Po '(?<=Driver Version: )\d+.\d+.\d+')

      # Hack alert: This url doesn't work with 430.XX family
      # TODO: figure out a better way to find the download path for the driver
      # Attempt to download several version namings of the file. If one download fails, it moves to the next
      declare -a versions_str_suffixes=("${NVIDIA_SMI_DRIVER_VERSION_LONG}" "${NVIDIA_SMI_DRIVER_VERSION_LONG}.00")
      for version_str_suffix in "${versions_str_suffixes[@]}"; do
          curl --fail http://us.download.nvidia.com/tesla/${NVIDIA_SMI_DRIVER_VERSION_LONG}/NVIDIA-Linux-x86_64-${version_str_suffix}.run \
               --output NVIDIA-Linux-x86_64-${NVIDIA_SMI_DRIVER_VERSION_LONG}.run \
              && break
      done

      chmod a+x *.run && ./*.run -x
      # put it to some TMP dir just in case we encounter read only /usr/lib/x86_64-linux-gnu in docker
      LIB_OF_DIR_PATH=$(mktemp -d)

      cp NVIDIA*/libnvidia-opticalflow* ${LIB_OF_DIR_PATH}
      ln -s ${LIB_OF_DIR_PATH}/libnvidia-opticalflow.so.${NVIDIA_SMI_DRIVER_VERSION_LONG} ${LIB_OF_DIR_PATH}/libnvidia-opticalflow.so.1
      ln -s ${LIB_OF_DIR_PATH}/libnvidia-opticalflow.so.1 ${LIB_OF_DIR_PATH}/libnvidia-opticalflow.so
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OF_DIR_PATH}
      rm -rf NVIDIA-Linux-*
  fi
}

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
