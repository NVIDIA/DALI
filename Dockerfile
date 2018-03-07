FROM nvdl.githost.io:4678/dgx/cuda:9.0-cudnn7.1-devel-ubuntu16.04--18.04

ARG PYVER=2.7

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      liblmdb-dev \
      autoconf \
      automake \
      libtool \
      nasm \
      python$PYVER \
      python$PYVER-dev \
      python$PYVER-numpy \
  && rm -rf /var/lib/apt/lists/*

# symlink so `python` works as expected everywhere
RUN ln -sf /usr/bin/python$PYVER /usr/bin/python
RUN ln -sf /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install future numpy setuptools

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd /opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr \
          -DWITH_CUDA=OFF -DWITH_1394=OFF \
          -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF -DWITH_IPP=OFF . && \
    make -j"$(nproc)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

# libjpeg-turbo
RUN JPEG_TURBO_VERSION=1.5.2 && \
    wget -q -O - https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_VERSION}.tar.gz | tar -xzf - && \
    cd libjpeg-turbo-${JPEG_TURBO_VERSION} && \
    autoreconf -fiv && \
    ./configure --enable-shared --prefix=/usr 2>&1 >/dev/null && \
    make -j"$(nproc)" install 2>&1 >/dev/null && \
    rm -rf /libjpeg-turbo-${JPEG_TURBO_VERSION}

# protobuf v3.5.1
RUN PROTOBUF_VERSION=3.5.1 && \
    wget -q -O - https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/usr 2>&1 > /dev/null && \
    make -j"$(nproc)" install 2>&1 > /dev/null && \
    rm -rf /protobuf-${PROTOBUF_VERSION}

WORKDIR /opt/ndll

COPY . .

RUN mkdir build && cd build && \
    cmake ../ -DCMAKE_INSTALL_PREFIX=/opt/ndll \
        -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_PYTHON=ON \
        -DBUILD_PROTOBUF=ON -DBUILD_LMDB=ON && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

ENV LD_LIBRARY_PATH /opt/ndll/build:$LD_LIBRARY_PATH

