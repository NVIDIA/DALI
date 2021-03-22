#!/bin/bash -xe

# LMDB
pushd third_party/lmdb/libraries/liblmdb/
patch -p3 < ${ROOT_DIR}/patches/Makefile-lmdb.patch
    CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC=${CC_COMP} CXX=${CXX_COMP} prefix=${INSTALL_PREFIX} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
