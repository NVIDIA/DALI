#!/bin/bash -xe

ROOT_DIR=$(pwd)
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

#zlib
pushd zlib                                                                        && \
CFLAGS="-fPIC" \
CXXFLAGS="-fPIC" \
CC=${CC_COMP} \
CXX=${CXX_COMP} \
./configure --prefix=${INSTALL_PREFIX}                                            && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"                                 && \
make install                                                                      && \
popd

# CMake
pushd CMake                                                                       && \
./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l)                   && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"                                 && \
make install                                                                      && \
popd

# protobuf, make two steps for cross compilation
pushd protobuf                                                                    && \
./autogen.sh                                                                      && \
./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" 2>&1 > /dev/null                && \
make install 2>&1 > /dev/null                                                     && \
make clean                                                                        && \
    ./autogen.sh && ./configure \
    CXXFLAGS="-fPIC ${EXTRA_PROTOBUF_FLAGS}" \
    CC=${CC_COMP} \
    CXX=${CXX_COMP} \
    ${HOST_ARCH_OPTION} \
    ${BUILD_ARCH_OPTION} \
    ${SYSROOT_ARG} \
    --with-protoc=/usr/local/bin/protoc \
    --prefix=${INSTALL_PREFIX} && make -j$(nproc) install                         && \
popd

# LMDB
pushd lmdb/libraries/liblmdb/                                                     && \
patch -p3 < ${ROOT_DIR}/Makefile-lmdb.patch                                       && \
  CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC=${CC_COMP} CXX=${CXX_COMP} prefix=${INSTALL_PREFIX} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"                                 && \
make install && \
popd

# libjpeg-turbo
pushd libjpeg-turbo/                                                              && \
echo "set(CMAKE_SYSTEM_NAME Linux)" > toolchain.cmake                             && \
echo "set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_TARGET_ARCH})" >> toolchain.cmake        && \
echo "set(CMAKE_C_COMPILER ${CC_COMP})" >> toolchain.cmake                        && \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DENABLE_SHARED=TRUE -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} . 2>&1 >/dev/null && \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" 2>&1 >/dev/null                 && \
make install 2>&1 >/dev/null                                                      && \
popd

# zstandard compression library
pushd zstd                                                                        && \
  CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC=${CC_COMP} CXX=${CXX_COMP} prefix=${INSTALL_PREFIX} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 >/dev/null         && \
popd

# libtiff
pushd libtiff                                                                     && \
./autogen.sh                                                                      && \
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
  --prefix=${INSTALL_PREFIX}                                                      && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"                                 && \
make install                                                                      && \
popd

# OpenJPEG
pushd openjpeg                                                                    && \
mkdir build && cd build                                                           && \
echo "set(CMAKE_SYSTEM_NAME Linux)" > toolchain.cmake                             && \
echo "set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_TARGET_ARCH})" >> toolchain.cmake        && \
echo "set(CMAKE_C_COMPILER ${CC_COMP})" >> toolchain.cmake                        && \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -DBUILD_CODEC=OFF \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ..                             && \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"                                 && \
make install                                                                      && \
popd

# OpenCV
pushd opencv                                                                      && \
patch -p1 < ${ROOT_DIR}/opencv-qnx.patch                                          && \
mkdir build && cd build                                                           && \
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DVIBRANTE_PDK:STRING=/ \
      -DCMAKE_TOOLCHAIN_FILE=$PWD/../platforms/${OPENCV_TOOLCHAIN_FILE} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -DBUILD_LIST=core,improc,imgcodecs \
      -DBUILD_SHARED_LIBS=OFF \
      -DWITH_CUDA=OFF \
      -DWITH_1394=OFF \
      -DWITH_IPP=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_GTK=OFF \
      -DBUILD_JPEG=OFF \
      -DWITH_JPEG=ON \
      -DBUILD_TIFF=OFF \
      -DWITH_TIFF=ON \
      -DBUILD_JASPER=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_PNG=ON \
      -DBUILD_opencv_cudalegacy=OFF \
      -DBUILD_opencv_stitching=OFF \
      -DWITH_TBB=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_PTHREADS_PF=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_java=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=OFF \
      -DWITH_PROTOBUF=OFF \
      -DWITH_FFMPEG=OFF \
      -DWITH_GSTREAMER=OFF \
      -DWITH_GSTREAMER_0_10=OFF \
      -DWITH_VTK=OFF \
      -DWITH_OPENEXR=OFF \
      -DINSTALL_C_EXAMPLES=OFF \
      -DINSTALL_TESTS=OFF \
      -DVIBRANTE=TRUE \
      -DWITH_CSTRIPES=OFF \
      VERBOSE=1 .. && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
make install && \
popd

if [ $WITH_FFMPEG -gt 0 ]; then
  # FFmpeg  https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-4.3.1.tar.bz2
  pushd FFmpeg                                                                    && \
  ./configure \
    --prefix=${INSTALL_PREFIX} \
    --disable-static \
    --disable-programs \
    --disable-doc \
    --disable-avdevice \
    --disable-swresample \
    --disable-swscale \
    --disable-postproc \
    --disable-w32threads \
    --disable-os2threads \
    --disable-dct \
    --disable-dwt \
    --disable-error-resilience \
    --disable-lsp \
    --disable-lzo \
    --disable-mdct \
    --disable-rdft \
    --disable-fft \
    --disable-faan \
    --disable-pixelutils \
    --disable-autodetect \
    --disable-iconv \
    --enable-shared \
    --enable-avformat \
    --enable-avcodec \
    --enable-avfilter \
    --disable-encoders \
    --disable-hwaccels \
    --disable-muxers \
    --disable-protocols \
    --enable-protocol=file \
    --disable-indevs \
    --disable-outdevs  \
    --disable-devices \
    --disable-filters \
    --disable-bsfs \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes           && \
  make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install               && \
  popd
fi

# flac
pushd flac                                                                        && \
./autogen.sh                                                                      && \
./configure \
  CFLAGS="-fPIC ${EXTRA_FLAC_FLAGS}" \
  CXXFLAGS="-fPIC ${EXTRA_FLAC_FLAGS}" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
  ${HOST_ARCH_OPTION} \
  --prefix=${INSTALL_PREFIX} \
  --disable-ogg                                                                   && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install                 && \
popd

# libogg
pushd ogg                                                                         && \
./autogen.sh                                                                      && \
./configure \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
  ${HOST_ARCH_OPTION} \
  --prefix=${INSTALL_PREFIX}                                                      && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install                 && \
popd

# libvorbis
# Install after libogg
pushd vorbis                                                                      && \
./autogen.sh                                                                      && \
./configure \
  CFLAGS="-fPIC" \
  CXXFLAGS="-fPIC" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
  ${HOST_ARCH_OPTION} \
  --prefix=${INSTALL_PREFIX}                                                      && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install                 && \
popd

# libsnd https://developer.download.nvidia.com/compute/redist/nvidia-dali/libsndfile-1.0.28.tar.gz
pushd libsndfile                                                                  && \
./autogen.sh                                                                      && \
./configure \
  CFLAGS="-fPIC ${EXTRA_LIBSND_FLAGS}" \
  CXXFLAGS="-fPIC ${EXTRA_LIBSND_FLAGS}" \
  CC=${CC_COMP} \
  CXX=${CXX_COMP} \
  ${HOST_ARCH_OPTION} \
  --prefix=${INSTALL_PREFIX}                                                      && \
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install                 && \
popd