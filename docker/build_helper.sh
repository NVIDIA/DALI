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
export BUILD_NVIMAGECODEC=${BUILD_NVIMAGECODEC:-ON}
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
export BUILD_CFITSIO=${BUILD_CFITSIO:-ON}
export BUILD_CVCUDA=${BUILD_CVCUDA:-ON}
export BUILD_CUFILE=${BUILD_CUFILE:-OFF}
export BUILD_NVCOMP=${BUILD_NVCOMP:-OFF}
export LINK_LIBCUDA=${LINK_LIBCUDA:-OFF}
export WITH_DYNAMIC_CUDA_TOOLKIT=${WITH_DYNAMIC_CUDA_TOOLKIT:-OFF}
export WITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG:-ON}
export WITH_DYNAMIC_CUFFT=${WITH_DYNAMIC_CUFFT:-ON}
export WITH_DYNAMIC_NPP=${WITH_DYNAMIC_NPP:-ON}
export WITH_DYNAMIC_NVIMGCODEC=${WITH_DYNAMIC_NVIMGCODEC:-ON}
export WITH_DYNAMIC_NVCOMP=${WITH_DYNAMIC_NVCOMP:-ON}
export STRIP_BINARY=${STRIP_BINARY:-OFF}
export VERBOSE_LOGS=${VERBOSE_LOGS:-OFF}
export WERROR=${WERROR:-ON}
export BUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}
export BUILD_WITH_LSAN=${BUILD_WITH_LSAN:-OFF}
export BUILD_WITH_UBSAN=${BUILD_WITH_UBSAN:-OFF}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-0}
export GIT_SHA=${GIT_SHA}
export DALI_TIMESTAMP=${DALI_TIMESTAMP}
export NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}
export CUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}
export WHL_PLATFORM_NAME=${WHL_PLATFORM_NAME:-manylinux_2_28_${ARCH}}
export WHL_OUTDIR=${WHL_OUTDIR:-/wheelhouse}
export WHL_COMPRESSION=${WHL_COMPRESSION:-YES}
export PATH=/usr/local/cuda/bin:${PATH}
export EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS}
export BUNDLE_PATH_PREFIX=${BUNDLE_PATH_PREFIX}
export TEST_BUNDLED_LIBS=${TEST_BUNDLED_LIBS:-YES}
export PYTHON_VERSIONS=${PYTHON_VERSIONS}
# use all available pythons

cmake ../ -DCMAKE_INSTALL_PREFIX=.                 \
      -DARCH=${ARCH}                               \
      -DCUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}     \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}       \
      -DBUILD_TEST=${BUILD_TEST}                   \
      -DBUILD_BENCHMARK=${BUILD_BENCHMARK}         \
      -DBUILD_NVTX=${BUILD_NVTX}                   \
      -DBUILD_PYTHON=${BUILD_PYTHON}               \
      -DBUILD_LMDB=${BUILD_LMDB}                   \
      -DBUILD_NVIMAGECODEC=${BUILD_NVIMAGECODEC}   \
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
      -DBUILD_CFITSIO=${BUILD_CFITSIO}             \
      -DBUILD_CUFILE=${BUILD_CUFILE}               \
      -DBUILD_NVCOMP=${BUILD_NVCOMP}               \
      -DBUILD_CVCUDA=${BUILD_CVCUDA}               \
      -DLINK_LIBCUDA=${LINK_LIBCUDA}               \
      -DWITH_DYNAMIC_CUDA_TOOLKIT=${WITH_DYNAMIC_CUDA_TOOLKIT} \
      -DWITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG} \
      -DWITH_DYNAMIC_CUFFT=${WITH_DYNAMIC_CUFFT}   \
      -DWITH_DYNAMIC_NPP=${WITH_DYNAMIC_NPP}       \
      -DWITH_DYNAMIC_NVIMGCODEC=${WITH_DYNAMIC_NVIMGCODEC} \
      -DWITH_DYNAMIC_NVCOMP=${WITH_DYNAMIC_NVCOMP} \
      -DVERBOSE_LOGS=${VERBOSE_LOGS}               \
      -DWERROR=${WERROR}                           \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN}         \
      -DBUILD_WITH_LSAN=${BUILD_WITH_LSAN}         \
      -DBUILD_WITH_UBSAN=${BUILD_WITH_UBSAN}       \
      -DPYTHON_VERSIONS=${PYTHON_VERSIONS}         \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} -DGIT_SHA=${GIT_SHA} \
      ${EXTRA_CMAKE_OPTIONS}
if [ "${WERROR}" = "ON" ]; then
    make -j lint
fi
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"

if [ "$BUILD_NVCOMP" = "ON" ] && ( [ "$WITH_DYNAMIC_NVCOMP" != "ON" ] || [ "$WITH_DYNAMIC_CUDA_TOOLKIT" != "ON" ] ); then
    export BUNDLE_NVCOMP=YES
else
    export BUNDLE_NVCOMP=NO
fi

bundle_wheel() {
    INPUT=$1
    STRIP=$2
    TEST_BUNDLED_LIBS=$3
    OUT_WHL_NAME=$4
    BUNDLE_PATH_PREFIX=$5
    ../dali/python/bundle-wheel.sh ${INPUT} ${STRIP} ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}" ${WHL_OUTDIR} ${WHL_COMPRESSION} ${BUNDLE_NVCOMP}
}


if [ "${BUILD_PYTHON}" = "ON" ]; then
    # use stored as a compression method to make it faster as bundle-wheel.sh need to repack anyway
    # call setup.py to avoid slow copy to tmp dir
    pushd dali/python
    python setup.py bdist_wheel \
        --verbose \
        --compression=stored \
        --python-tag=py3 \
        --plat-name=${WHL_PLATFORM_NAME}
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
        pids[0]=$!
        bundle_wheel nvidia_dali[_-]*.whl YES ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}" &
        pids[1]=$!
        for pid in ${pids[*]}; do
            wait $pid
        done
    else
        bundle_wheel nvidia_dali[_-]*.whl NO ${TEST_BUNDLED_LIBS} ${OUT_WHL_NAME} "${BUNDLE_PATH_PREFIX}"
    fi
fi
