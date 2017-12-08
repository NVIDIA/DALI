FROM nvdl.githost.io:4678/dgx/cuda:9.0-cudnn7-devel-ubuntu16.04--18.01

ARG PYVER=2.7

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      liblmdb-dev \
      libprotobuf-dev \
      protobuf-compiler \
      autoconf \
      automake \
      nasm \
  && rm -rf /var/lib/apt/lists/*

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

WORKDIR /opt/ndll

COPY . .

RUN mkdir build && cd build && \
		cmake ../ -DCMAKE_INSTALL_PREFIX=/opt/ndll \
				-DBUILD_TEST=ON -DBUILD_BENCHMARK=ON && \
		make -j"$(nproc)" && \
		ldconfig

ENV LD_LIBRARY_PATH /opt/ndll/build:$LD_LIBRARY_PATH

