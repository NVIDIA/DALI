#!/bin/bash -xe

export ROOT_DIR=$(pwd)
export SCRIPT_DIR=$(cd $(dirname $0) && pwd)
export INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}
export CC_COMP=${CC_COMP:-gcc}
export CXX_COMP=${CXX_COMP:-g++}
echo ${INSTALL_PREFIX}
echo ${CC_COMP}
echo ${CXX_COMP}
echo ${CMAKE_TARGET_ARCH}
echo ${BUILD_ARCH_OPTION}
echo ${HOST_ARCH_OPTION}
echo ${SYSROOT_ARG}
echo ${WITH_FFMPEG}
echo ${EXTRA_PROTOBUF_FLAGS}
echo ${OPENCV_TOOLCHAIN_FILE}
echo ${EXTRA_FLAC_FLAGS}
echo ${EXTRA_LIBSND_FLAGS}

PACKAGE_LIST=(
    "zlib"
    "cmake"
    "protobuf"
    "lmdb"
    "libjpeg-turbo"
    "zstd"
    "openjpeg"
    "libtiff"
    "opencv"
    "ffmpeg"
    "libflac"
    "libogg"
    "libvorbis" # Install after libogg
    "libsnd"
)

for PACKAGE in "${PACKAGE_LIST[@]}"; do
    ${SCRIPT_DIR}/build_${PACKAGE}.sh
done
