#!/bin/bash -xe

# libvorbis
pushd third_party/vorbis
./autogen.sh
./configure \
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
    ${HOST_ARCH_OPTION} \
    --prefix=${INSTALL_PREFIX}
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
