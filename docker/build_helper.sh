#!/bin/bash

# Stop at any error, show all commands
set -ex

usage="ENV1=VAL1 ENV2=VAL2 [...] $(basename "$0") [-h] -- this DALI build helper mean to run from the docker environment.
Please don't call it directly.

where:
    -h  show this help text"

while getopts 'h' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export BUILD_TEST=${BUILD_TEST:-ON}
export BUILD_BENCHMARK=${BUILD_BENCHMARK:-ON}
export BUILD_NVTX=${BUILD_NVTX:-OFF}
export BUILD_PYTHON=${BUILD_PYTHON:-ON}
export BUILD_LMDB=${BUILD_LMDB:-ON}
export BUILD_TENSORFLOW=${BUILD_TENSORFLOW:-OFF}
export BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}
export BUILD_NVJPEG=${BUILD_NVJPEG:-ON}
export BUILD_NVOF=${BUILD_NVOF:-ON}
export BUILD_NVDEC=${BUILD_NVDEC:-ON}
export BUILD_NVML=${BUILD_NVML:-ON}
export WERROR=${WERROR:-ON}
export BUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-0}
export GIT_SHA=${GIT_SHA}
export DALI_TIMESTAMP=${DALI_TIMESTAMP}
export NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}

LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}" && \
cmake ../ -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
      -DBUILD_TEST=${BUILD_TEST} -DBUILD_BENCHMARK=${BUILD_BENCHMARK} \
      -DBUILD_NVTX=${BUILD_NVTX} -DBUILD_PYTHON=${BUILD_PYTHON} \
      -DBUILD_LMDB=${BUILD_LMDB} -DBUILD_TENSORFLOW=${BUILD_TENSORFLOW}  \
      -DBUILD_JPEG_TURBO=${BUILD_JPEG_TURBO} -DBUILD_NVJPEG=${BUILD_NVJPEG} \
      -DBUILD_NVOF=${BUILD_NVOF} -DBUILD_NVDEC=${BUILD_NVDEC} \
      -DBUILD_NVML=${BUILD_NVML} \
      -DWERROR=${WERROR} \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN} \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} -DGIT_SHA=${GIT_SHA}
if [ "${WERROR}" = "ON" ]; then
    make -j lint
fi
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"

if [ "${BUILD_PYTHON}" = "ON" ]; then \
    pip wheel -v dali/python \
        --build-option --python-tag=$(basename /opt/python/cp${PYV}-*) \
        --build-option --plat-name=manylinux1_x86_64 \
        --build-option --build-number=${NVIDIA_BUILD_ID}
    ../dali/python/bundle-wheel.sh nvidia_dali[_-]*.whl
    export UNZIP_PATH="$(mktemp -d)"
    unzip /wheelhouse/nvidia_dali*.whl -d $UNZIP_PATH
    python ../tools/test_bundled_libs.py $(find $UNZIP_PATH -iname *.so* | tr '\n' ' ')
    rm -rf $UNZIP_PATH
fi
