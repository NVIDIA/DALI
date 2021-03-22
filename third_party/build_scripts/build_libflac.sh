#!/bin/bash -xe

# flac
pushd third_party/flac
./autogen.sh
./configure \
  CFLAGS="-fPIC ${EXTRA_FLAC_FLAGS}" \
  CXXFLAGS="-fPIC ${EXTRA_FLAC_FLAGS}" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
  ${HOST_ARCH_OPTION} \
  --prefix=${INSTALL_PREFIX} \
  --disable-ogg
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
make install
popd
