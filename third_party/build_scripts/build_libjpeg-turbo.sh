#!/bin/bash -xe

# libjpeg-turbo
pushd third_party/libjpeg-turbo/
echo "set(CMAKE_SYSTEM_NAME Linux)" > toolchain.cmake
echo "set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_TARGET_ARCH})" >> toolchain.cmake
echo "set(CMAKE_C_COMPILER ${CC_COMP})" >> toolchain.cmake
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DENABLE_SHARED=TRUE -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} . 2>&1 >/dev/null
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" 2>&1 >/dev/null
make install 2>&1 >/dev/null
popd
