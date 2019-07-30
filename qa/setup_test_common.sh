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

put_optflow_libs() {
  # Docker v1 doesn't expose libnvidia-opticalflow.so from the host so we need to manually put it there
  # workaround for the CI
  if [[ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so ]]; then
      NVIDIA_SMI_DRIVER_VERSION_LONG=$(nvidia-smi | grep -Po '(?<=Driver Version: )\d+.\d+.\d+')
      curl http://us.download.nvidia.com/tesla/${NVIDIA_SMI_DRIVER_VERSION_LONG}/NVIDIA-Linux-x86_64-${NVIDIA_SMI_DRIVER_VERSION_LONG}.run --output NVIDIA-Linux-x86_64-${NVIDIA_SMI_DRIVER_VERSION_LONG}.run
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
