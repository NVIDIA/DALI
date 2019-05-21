FROM nvidia/cuda:10.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    unzip \
    git \
    rsync \
    libjpeg-dev \
    dh-autoreconf \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

ENV REPO_DEBS="cuda-repo-ubuntu1604-10-0-local-10.0.117-410.38_1.0-1_amd64.deb"
ENV CUDA_CROSS_VERSION=10-0
ENV CUDA_CROSS_PACKAGES="cublas cudart cufft curand cusolver cusparse driver misc-headers npp"
#nvml nvrtc nvgraph"

RUN wget https://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/4.1.1/xddsn.im/JetPackL4T_4.1.1_b57/16.04/cuda-repo-ubuntu1604-10-0-local-10.0.117-410.38_1.0-1_amd64.deb && \
    dpkg -i $REPO_DEBS && \
    echo "for i in \$CUDA_CROSS_PACKAGES; do echo \"cuda-\$i-cross-aarch64-\${CUDA_CROSS_VERSION}\";done" | bash > /tmp/cuda-packages.txt && \
    apt-get update \
   && apt-get install -y $(cat /tmp/cuda-packages.txt) \
   && rm -rf /var/lib/apt/lists/* \
   && rm -rf /tmp/cuda-packages.txt

# Boost
RUN BOOST_VERSION=1_66_0 \
   && cd /usr/local \
   && curl -L https://dl.bintray.com/boostorg/release/1.66.0/source/boost_${BOOST_VERSION}.tar.gz | tar -xzf - \
   && ln -s ../boost_${BOOST_VERSION}/boost include/boost

# CMake
RUN CMAKE_VERSION=3.11 && \
    CMAKE_BUILD=3.11.0 && \
    curl -L https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz | tar -xzf - && \
    cd /cmake-${CMAKE_BUILD} && \
    ./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l) && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /cmake-${CMAKE_BUILD}

# protobuf v3.5.1
ENV PROTOBUF_VERSION=3.5.1
RUN curl -L https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd /protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 > /dev/null

RUN cd /protobuf-${PROTOBUF_VERSION} && make clean \
    ./autogen.sh && ./configure \
    CXXFLAGS="-fPIC" \
    CC=aarch64-linux-gnu-gcc \
    CXX=aarch64-linux-gnu-g++ \
      --host=aarch64-unknown-linux-gnu \
      --with-protoc=/usr/local/bin/protoc \
      --prefix=/usr/aarch64-linux-gnu/ && make -j$(nproc) install && \
    rm -rf /protobuf-${PROTOBUF_VERSION}


ENV JPEG_TURBO_VERSION=1.5.3
RUN curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_VERSION}.tar.gz | tar -xzf - && \
    cd /libjpeg-turbo-${JPEG_TURBO_VERSION} && \
    autoreconf -fiv && \
    ./configure \
      --disable-shared \
      CFLAGS="-fPIC" \
      CXXFLAGS="-fPIC" \
      CC=aarch64-linux-gnu-gcc \
      CXX=aarch64-linux-gnu-g++ \
      --host=aarch64-unknown-linux-gnu \
      --prefix=/usr/aarch64-linux-gnu/ && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /libjpeg-turbo-${JPEG_TURBO_VERSION}

# OpenCV
ENV OPENCV_VERSION=3.4.3
RUN curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd /opencv-${OPENCV_VERSION} && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE=$PWD/../platforms/linux/aarch64-gnu.toolchain.cmake \
          -DCMAKE_INSTALL_PREFIX=/usr/aarch64-linux-gnu/ \
          -DBUILD_SHARED_LIBS=OFF \
          -DBUILD_LIST=core,improc,imgcodecs \
          -DBUILD_PNG=ON \
          -DBUILD_TIFF=OFF \
          -DBUILD_TBB=OFF \
          -DBUILD_WEBP=OFF \
          -DBUILD_JPEG=OFF \
          -DWITH_JPEG=ON \
          -DBUILD_JASPER=OFF \
          -DBUILD_ZLIB=ON \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_FFMPEG=ON \
          -DBUILD_opencv_java=OFF \
          -DBUILD_opencv_python2=OFF \
          -DBUILD_opencv_python3=OFF \
          -DENABLE_NEON=OFF \
          -DWITH_PROTOBUF=OFF \
          -DWITH_PTHREADS_PF=OFF \
          -DWITH_OPENCL=OFF \
          -DWITH_OPENMP=OFF \
          -DWITH_FFMPEG=OFF \
          -DWITH_GSTREAMER=OFF \
          -DWITH_GSTREAMER_0_10=OFF \
          -DWITH_CUDA=OFF \
          -DWITH_GTK=OFF \
          -DWITH_VTK=OFF \
          -DWITH_TBB=OFF \
          -DWITH_1394=OFF \
          -DWITH_OPENEXR=OFF \
          -DINSTALL_C_EXAMPLES=OFF \
          -DINSTALL_TESTS=OFF \
          -DVIBRANTE=TRUE \
          VERBOSE=1 ../  && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

VOLUME /dali

WORKDIR /dali


ENV PATH=/usr/local/cuda-10.0/bin:$PATH

ARG DALI_BUILD_DIR=build_aarch64_linux

WORKDIR /dali/${DALI_BUILD_DIR}

CMD cmake \
  -DWERROR=ON \
  -DCMAKE_TOOLCHAIN_FILE:STRING="$PWD/../platforms/aarch64-linux/aarch64-linux.toolchain.cmake" \
  -DCMAKE_COLOR_MAKEFILE=ON \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DARCH=aarch64-linux \
  -DCUDA_HOST=/usr/local/cuda-10.0 \
  -DCUDA_TARGET=/usr/local/cuda-10.0/targets/aarch64-linux \
  -DBUILD_TEST=ON \
  -DBUILD_BENCHMARK=OFF \
  -DBUILD_NVTX=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_LMDB=OFF \
  -DBUILD_TENSORFLOW=OFF \
  -DBUILD_JPEG_TURBO=ON \
  -DBUILD_NVJPEG=OFF \
  -DBUILD_NVOF=OFF \
  -DBUILD_NVDEC=OFF \
  -DBUILD_NVML=OFF \
  ..  && \
  make -j"$(grep ^processor /proc/cpuinfo | wc -l)"