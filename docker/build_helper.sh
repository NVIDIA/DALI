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

export ARCH=${ARCH}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export BUILD_TEST=${BUILD_TEST:-ON}
export BUILD_BENCHMARK=${BUILD_BENCHMARK:-ON}
# use a default value as it differs for CUDA 9.x and CUDA 10.x
export BUILD_NVTX=${BUILD_NVTX}
export BUILD_PYTHON=${BUILD_PYTHON:-ON}
export BUILD_LMDB=${BUILD_LMDB:-ON}
export BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}
export BUILD_OPENCV=${BUILD_OPENCV:-ON}
export BUILD_PROTOBUF=${BUILD_PROTOBUF:-ON}
export BUILD_NVJPEG=${BUILD_NVJPEG:-ON}
# use a default value as it differs for CUDA 10 and CUDA 11.x
export BUILD_NVJPEG2K=${BUILD_NVJPEG2K:-ON}
export BUILD_LIBTIFF=${BUILD_LIBTIFF:-ON}
export BUILD_NVOF=${BUILD_NVOF:-ON}
export BUILD_NVDEC=${BUILD_NVDEC:-ON}
export BUILD_LIBSND=${BUILD_LIBSND:-ON}
export BUILD_LIBTAR=${BUILD_LIBTAR:-ON}
export BUILD_NVML=${BUILD_NVML:-ON}
export BUILD_FFTS=${BUILD_FFTS:-ON}
export BUILD_CUFILE=${BUILD_CUFILE-OFF}
export LINK_LIBCUDA=${LINK_LIBCUDA:-OFF}
export STRIP_BINARY=${STRIP_BINARY:-OFF}
export VERBOSE_LOGS=${VERBOSE_LOGS:-OFF}
export WERROR=${WERROR:-ON}
export BUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-0}
export GIT_SHA=${GIT_SHA}
export DALI_TIMESTAMP=${DALI_TIMESTAMP}
export NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}
export CUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}
export WHL_PLATFORM_NAME=${WHL_PLATFORM_NAME:-manylinux2014_${ARCH}}
export PATH=/usr/local/cuda/bin:${PATH}
export EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS}
export BUNDLE_PATH_PREFIX=${BUNDLE_PATH_PREFIX}
export TEST_BUNDLED_LIBS=${TEST_BUNDLED_LIBS:-YES}
# use all avialble pythons

LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}" && \
cmake ../ -DCMAKE_INSTALL_PREFIX=.                 \
      -DARCH=${ARCH}                               \
      -DCUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}     \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}       \
      -DBUILD_TEST=${BUILD_TEST}                   \
      -DBUILD_BENCHMARK=${BUILD_BENCHMARK}         \
      -DBUILD_NVTX=${BUILD_NVTX}                   \
      -DBUILD_PYTHON=${BUILD_PYTHON}               \
      -DBUILD_LMDB=${BUILD_LMDB}                   \
      -DBUILD_JPEG_TURBO=${BUILD_JPEG_TURBO}       \
      -DBUILD_OPENCV=${BUILD_OPENCV}               \
      -DBUILD_PROTOBUF=${BUILD_PROTOBUF}           \
      -DBUILD_NVJPEG=${BUILD_NVJPEG}               \
      -DBUILD_NVJPEG2K=${BUILD_NVJPEG2K}           \
      -DBUILD_LIBTIFF=${BUILD_LIBTIFF}             \
      -DBUILD_NVOF=${BUILD_NVOF}                   \
      -DBUILD_NVDEC=${BUILD_NVDEC}                 \
      -DBUILD_LIBSND=${BUILD_LIBSND}               \
      -DBUILD_NVML=${BUILD_NVML}                   \
      -DBUILD_FFTS=${BUILD_FFTS}                   \
      -DBUILD_CUFILE=${BUILD_CUFILE}               \
      -DLINK_LIBCUDA=${LINK_LIBCUDA}               \
      -DVERBOSE_LOGS=${VERBOSE_LOGS}               \
      -DWERROR=${WERROR}                           \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN}         \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} -DGIT_SHA=${GIT_SHA} \
      ${EXTRA_CMAKE_OPTIONS}
if [ "${WERROR}" = "ON" ]; then
    make -j lint
fi
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"


bundle_wheel() {
    INPUT=$1
    STRIP=$2
    TEST_BUNDLED_LIBS=$3
    OUT_WHL_NAME=$4
    BUNDLE_PATH_PREFIX=$5
    ../dali/python/bundle-wheel.sh ${INPUT} ${STRIP} ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}"
}


if [ "${BUILD_PYTHON}" = "ON" ]; then
    # use stored as a compression method to make it faster as bundle-wheel.sh need to repack anyway
    # call setup.py to avoid slow copy to tmp dir
    pushd dali/python
    python setup.py bdist_wheel \
        --verbose \
        --compression=stored \
        --python-tag=py3-none \
        --plat-name=${WHL_PLATFORM_NAME} \
        --build-number=${NVIDIA_BUILD_ID}
    popd
    mv dali/python/dist/*.whl ./

    OUT_WHL_NAME=$(echo nvidia_dali[_-]*.whl)
    OUT_DEBUG_WHL_NAME=${OUT_WHL_NAME%.*}_debug.whl

    if [ "${STRIP_BINARY}" = "ON" ]; then
        # rerun all things involving patchelf on striped binary
        # we cannot strip after patchelf as according to the documentation
        ###
        ### The `strip' command from binutils generated broken executables when
        ### applied to the output of patchelf (if `--set-rpath' or
        ### `--set-interpreter' with a larger path than the original is used).
        ### This appears to be a bug in binutils
        ### (http://bugs.strategoxt.org/browse/NIXPKGS-85).
        bundle_wheel nvidia_dali[_-]*.whl NO NO ${OUT_DEBUG_WHL_NAME} "${BUNDLE_PATH_PREFIX}" &
        bundle_wheel nvidia_dali[_-]*.whl YES ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}" &
        wait
    else
        bundle_wheel nvidia_dali[_-]*.whl NO ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}"
    fi
fi
