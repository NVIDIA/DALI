#!/bin/bash -xe

# protobuf, make two steps for cross compilation if needed
pushd third_party/protobuf
./autogen.sh
./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" 2>&1 > /dev/null
make install 2>&1 > /dev/null
# only when cross compiling
if [ "${CC_COMP}" != "gcc" ]; then
  make clean
  ./autogen.sh
  ./configure \
      CXXFLAGS="-fPIC ${EXTRA_PROTOBUF_FLAGS}" \
      CC=${CC_COMP} \
      CXX=${CXX_COMP} \
      ${HOST_ARCH_OPTION} \
      ${BUILD_ARCH_OPTION} \
      ${SYSROOT_ARG} \
      --with-protoc=/usr/local/bin/protoc \
      --prefix=${INSTALL_PREFIX}
  make -j$(nproc) install
fi
popd
