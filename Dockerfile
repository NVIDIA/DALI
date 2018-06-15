#########################################################################################
##  Stage 1: build DALI wheels using manylinux1 (CentOS 5 derivative)
#########################################################################################
FROM quay.io/pypa/manylinux1_x86_64 AS builder

# Install yum Dependencies
RUN yum install -y zip

# Remove unnecessary Python version (narrow unicode Py2.7) from base container (must do this after yum)
RUN rm -rf /opt/python/cp27-cp27m /opt/_internal/cpython-2.7.15-ucs2 && \
    cp /opt/python/cp27-cp27mu/bin/python2.7 /usr/bin && \
    cp /opt/python/cp27-cp27mu/bin/python /usr/bin

# Install python dependencies
RUN for PYBIN in /opt/python/*/bin; do \
        "${PYBIN}/pip" install future numpy setuptools tensorflow-gpu; \
    done

# Boost
RUN BOOST_VERSION=1.66.0 && \
    cd /usr/local && \
    curl -L https://dl.bintray.com/boostorg/release/1.66.0/source/boost_${BOOST_VERSION//./_}.tar.gz | tar -xzf - && \
    ln -s ../boost_${BOOST_VERSION//./_}/boost include/boost

# CMake
RUN CMAKE_VERSION=3.11 && \
    CMAKE_BUILD=3.11.3 && \
    curl -L https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz | tar -xzf - && \
    cd /cmake-${CMAKE_BUILD} && \
    ./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l) && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /cmake-${CMAKE_BUILD}

# protobuf v3.5.1
RUN PROTOBUF_VERSION=3.5.1 && \
    curl -L https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd /protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local 2>&1 > /dev/null && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 > /dev/null && \
    rm -rf /protobuf-${PROTOBUF_VERSION}

# LMDB
RUN LMDB_VERSION=0.9.22 && \
    git clone -b LMDB_${LMDB_VERSION} --single-branch https://github.com/LMDB/lmdb && \
    cd /lmdb/libraries/liblmdb && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /lmdb

# OpenCV
RUN OPENCV_VERSION=3.1.0 && \
    curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd /opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DWITH_CUDA=OFF -DWITH_1394=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_GTK=OFF \
          -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
          -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF . && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

# libjpeg-turbo
#
# Note: the preceding OpenCV installation intentionally does NOT use libjpeg-turbo.
# DALI will directly call libjpeg-turbo first, and if it fails, DALI will fall back
# to OpenCV, which in turn will call its bundled (built-from-source) libjpeg.
# To be extra sure OpenCV doesn't pick up libjpeg-turbo (in which case we'd have no
# fallback), libjpeg-turbo is built and installed _after_ OpenCV.
#
RUN JPEG_TURBO_VERSION=1.5.2 && \
    curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/${JPEG_TURBO_VERSION}.tar.gz | tar -xzf - && \
    cd /libjpeg-turbo-${JPEG_TURBO_VERSION} && \
    autoreconf -fiv && \
    ./configure --enable-shared --prefix=/usr/local 2>&1 >/dev/null && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 >/dev/null && \
    rm -rf /libjpeg-turbo-${JPEG_TURBO_VERSION}

# CUDA
RUN CUDA_VERSION=9.0 && \
    CUDA_BUILD=9.0.176_384.81 && \
    curl -LO https://developer.nvidia.com/compute/cuda/${CUDA_VERSION}/Prod/local_installers/cuda_${CUDA_BUILD}_linux-run && \
    chmod +x cuda_${CUDA_BUILD}_linux-run && \
    ./cuda_${CUDA_BUILD}_linux-run --silent --no-opengl-libs --toolkit && \
    rm -f cuda_${CUDA_BUILD}_linux-run

# NVJPEG
RUN NVJPEG_VERSION=9.0.450-24313934 && \
    curl -L http://sqrl/dldata/nvjpeg/cuda-linux64-nvjpeg-${NVJPEG_VERSION}.tar.gz | tar -xzf - && \
    cd /cuda-linux64-nvjpeg/ && \
    mv lib64/libnvjpeg.so* /usr/local/lib/ && \
    mv include/nvjpeg.h /usr/local/include/ && \
    rm -rf /cuda-linux64-nvjpeg

WORKDIR /opt/dali

COPY . .

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} && \
    for PYVER in $(ls /opt/python); do \
      ( \
        set -e; \
        PYTHONPATH="/opt/python/${PYVER}" && \
        PYBIN="${PYTHONPATH}/bin" && \
        PYLIB="${PYTHONPATH}/lib" && \
        PATH="${PYBIN}:${PATH}" && \
        mkdir build-${PYVER} && \
        pushd build-${PYVER} && \
        LD_LIBRARY_PATH="${PYLIB}:${PWD}:${LD_LIBRARY_PATH}" && \
        cmake ../ -DCMAKE_INSTALL_PREFIX=. \
            -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_PYTHON=ON \
            -DBUILD_LMDB=ON -DBUILD_TENSORFLOW=ON && \
        make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
        popd \
      ); \
    done

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-0}

RUN for PYVER in $(ls /opt/python); do \
      ( \
        set -e; \
        PYTHONPATH="/opt/python/${PYVER}" && \
        PYBIN="${PYTHONPATH}/bin" && \
        PYLIB="${PYTHONPATH}/lib" && \
        PATH="${PYBIN}:${PATH}" && \
        pushd build-${PYVER} && \
        LD_LIBRARY_PATH="${PYLIB}:${PWD}:${LD_LIBRARY_PATH}" && \
        "${PYBIN}/pip" wheel -v ndll/python \
            --build-option --python-tag=${PYVER} \
            --build-option --plat-name=manylinux1_x86_64 \
            --build-option --build-number=${NVIDIA_BUILD_ID} && \
        ../ndll/python/bundle-wheel.sh nvidia_dali-*.whl && \
      ); \
    done

#########################################################################################
##  Stage 2: install Dali on top of the NGC CUDA+cuDNN base image
#########################################################################################
FROM gitlab-dl.nvidia.com:5005/dgx/cuda:9.0-cudnn7.1-devel-ubuntu16.04--18.07 AS dlbase

ARG PYVER=2.7
ARG PYV=27

RUN apt-get update && apt-get install -y --no-install-recommends \
        doxygen \
        python$PYVER \
        python$PYVER-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING utf-8
ENV LC_ALL C.UTF-8
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python`echo $PYVER | cut -c1-1` && \
    ln -s /usr/bin/python$PYVER /usr/bin/python && \
    ln -s /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

# If installing multiple pips, install pip2 last so that pip == pip2 when done.
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

COPY Acknowledgements.txt /opt/dali/
COPY COPYRIGHT   /opt/dali/
COPY Doxyfile    /opt/dali/
COPY LICENSE     /opt/dali/
COPY README.md  /opt/dali/
COPY docs       /opt/dali/
COPY examples   /opt/dali/
COPY scripts    /opt/dali/
COPY tools      /opt/dali/

COPY --from=builder /wheelhouse/nvidia_dali-*${PYV}-* /opt/dali/

RUN pip install /opt/dali/*.whl

RUN cd /opt/dali && doxygen Doxyfile
