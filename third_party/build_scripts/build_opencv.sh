#!/bin/bash -xe

# OpenCV
pushd third_party/opencv
patch -p1 < ${ROOT_DIR}/patches/opencv-qnx.patch
mkdir build
cd build
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
      VERBOSE=1 ..
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install
make install
popd
