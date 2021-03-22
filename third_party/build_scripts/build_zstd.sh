#!/bin/bash -xe

# zstandard compression library
pushd third_party/zstd
   CFLAGS="-fPIC" \
   CXXFLAGS="-fPIC" \
   CC=${CC_COMP} \
   CXX=${CXX_COMP} \
   prefix=${INSTALL_PREFIX} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 >/dev/null
popd
