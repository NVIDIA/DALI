#!/bin/bash -xe

# libsnd https://developer.download.nvidia.com/compute/redist/nvidia-dali/libsndfile-1.0.28.tar.gz
pushd third_party/libsndfile
./autogen.sh
./configure \
    CFLAGS="-fPIC ${EXTRA_LIBSND_FLAGS}" \
    CXXFLAGS="-fPIC ${EXTRA_LIBSND_FLAGS}" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
    ${HOST_ARCH_OPTION} \
    --prefix=${INSTALL_PREFIX}
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
