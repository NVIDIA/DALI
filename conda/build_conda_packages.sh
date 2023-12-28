#!/bin/bash

set -o xtrace
set -e

# Adding conda-forge channel for dependencies
conda config --add channels conda-forge
conda config --add channels nvidia
conda config --add channels local

CONDA_BUILD_OPTIONS="--exclusive-config-file config/conda_build_config.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

export CUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export BUILD_TEST=${BUILD_TEST:-ON}
export BUILD_BENCHMARK=${BUILD_BENCHMARK:-ON}
export BUILD_FUZZING=${BUILD_FUZZING:-OFF}
export BUILD_NVTX=${BUILD_NVTX}
export BUILD_LMDB=${BUILD_LMDB:-ON}
export BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}
export BUILD_NVJPEG=${BUILD_NVJPEG:-ON}
export BUILD_LIBTIFF=${BUILD_LIBTIFF:-ON}
export BUILD_LIBSND=${BUILD_LIBSND:-ON}
export BUILD_LIBTAR=${BUILD_LIBTAR:-ON}
export BUILD_FFTS=${BUILD_FFTS:-ON}
export BUILD_CFITSIO=${BUILD_CFITSIO:-ON}
export BUILD_NVOF=${BUILD_NVOF:-ON}
export BUILD_NVDEC=${BUILD_NVDEC:-ON}
export BUILD_NVML=${BUILD_NVML:-ON}
export WITH_DYNAMIC_CUDA_TOOLKIT=${WITH_DYNAMIC_CUDA_TOOLKIT:-OFF}
export WITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG:-ON}
export WITH_DYNAMIC_CUFFT=${WITH_DYNAMIC_CUFFT:-ON}
export WITH_DYNAMIC_NPP=${WITH_DYNAMIC_NPP:-ON}
export VERBOSE_LOGS=${VERBOSE_LOGS:-OFF}
export WERROR=${WERROR:-ON}
export BUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}
export BUILD_WITH_LSAN=${BUILD_WITH_LSAN:-OFF}
export BUILD_WITH_UBSAN=${BUILD_WITH_UBSAN:-OFF}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-0}
export GIT_SHA=${GIT_SHA}
export DALI_TIMESTAMP=${DALI_TIMESTAMP}
export NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export DALI_CONDA_BUILD_VERSION=$(cat ../VERSION)$(if [ "${NVIDIA_DALI_BUILD_FLAVOR}" != "" ]; then \
                                                     echo .${NVIDIA_DALI_BUILD_FLAVOR}.${DALI_TIMESTAMP}; \
                                                   fi)
export CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1.\2/')
export CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
# when building for any version >= 11.0 use CUDA compatibility mode and claim it is a CUDA 110 package
if [ "${CUDA_VERSION/./}" -ge 110 ]; then
  export CUDA_VERSION="${CUDA_VERSION%?}0"
fi

# Build custom OpenCV first, as DALI requires only bare OpenCV without many features and dependencies
conda mambabuild ${CONDA_BUILD_OPTIONS} third_party/dali_opencv/recipe

# Build custom FFmpeg, as DALI requires only bare FFmpeg without many features and dependencies
# but wiht mpeg4_unpack_bframes enabled
conda mambabuild ${CONDA_BUILD_OPTIONS} third_party/dali_ffmpeg/recipe

# Building DALI core package
conda mambabuild ${CONDA_BUILD_OPTIONS} dali_core/recipe

# Building DALI python bindings package
conda mambabuild ${CONDA_BUILD_OPTIONS} --variants="{python: [3.8, 3.9, 3.10, 3.11, 3.12]}" dali_bindings/recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/nvidia-dali*.tar.bz2 artifacts
