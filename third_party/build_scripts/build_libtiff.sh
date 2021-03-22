#!/bin/bash -xe

# libtiff
pushd third_party/libtiff
./autogen.sh
./configure \
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
    ${HOST_ARCH_OPTION} \
    --with-zstd-include-dir=${INSTALL_PREFIX}/include \
    --with-zstd-lib-dir=${INSTALL_PREFIX}/lib         \
    --with-zlib-include-dir=${INSTALL_PREFIX}/include \
    --with-zlib-lib-dir=${INSTALL_PREFIX}/lib         \
    --prefix=${INSTALL_PREFIX}
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
