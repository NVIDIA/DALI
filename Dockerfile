# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Tag: cuda:10.0-cudnn7-devel-ubuntu16.04
# Env: CUDA_VERSION=9.2.148
# Env: NCCL_VERSION=2.4.8
# Env: CUDNN_VERSION=7.6.3.30
# Ubuntu 16.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

USER root:root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH "/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL


# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    iproute2 && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    git \
    wget \
    unzip \
    libjpeg-dev \
    libpng-dev \
    dh-autoreconf \
    ca-certificates \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    cpio \
    yasm && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Get Conda-ified Python.
ENV CONDA_HOME /opt/conda
RUN echo 'export PATH=${CONDA_HOME}/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    sh ~/anaconda.sh -b -p ${CONDA_HOME} && \
    rm ~/anaconda.sh
ENV PATH ${CONDA_HOME}/bin:$PATH

# Install general libraries
RUN conda install -y python=3.7 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install -y mkl-include cmake cffi typing cython
RUN conda install -y -c mingfeima mkldnn

# Open-MPI installation
ENV OPENMPI_VERSION 3.1.2
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -L https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install pytorch 
RUN conda install -y pytorch==1.3.1 torchvision cudatoolkit=10.0 -c pytorch

# Set CUDA_ROOT
ENV CUDA_HOME="/usr/local/cuda"

# pip packages
RUN pip install mmcv pycocotools terminaltables future tensorboard termcolor

# Install apex
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    rm -rf /tmp/unique_for_apex

# Install opencv
WORKDIR /tmp/unique_for_cv2
RUN OPENCV_VERSION=4.3.0 && \
    wget --quiet https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && cd opencv-${OPENCV_VERSION} && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${CONDA_HOME} .. && \
    make -j $(nproc) && make install && \
    rm -rf /tmp/unique_for_cv2

# Install libsnd
WORKDIR /tmp/unique_for_libsnd
RUN LIBSND_VERSION=1.0.28 && \
    curl -L https://developer.download.nvidia.com/compute/redist/nvidia-dali/libsndfile-${LIBSND_VERSION}.tar.gz | tar -xzf - && \
    cd libsndfile-${LIBSND_VERSION} && \
    ./configure && make -j $(nproc) && make install && \
    rm -rf /tmp/unique_for_libsnd

# Install FFmpeg
WORKDIR /tmp/unique_for_FFmpeg
RUN FFMPEG_VERSION=4.2.2 && \
    curl -L https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-${FFMPEG_VERSION}.tar.bz2 | tar -xjf - && \
    cd ffmpeg-${FFMPEG_VERSION} && \
    ./configure --prefix=/usr/local --disable-static --disable-all --disable-autodetect --disable-iconv --enable-shared --enable-avformat --enable-avcodec --enable-avfilter --enable-protocol=file --enable-demuxer=mov,matroska,avi --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes  && \
    make -j $(nproc) && make install && \
    rm -rf /tmp/unique_for_FFmpeg

# Boost
RUN BOOST_VERSION=1_66_0 \
   && cd /usr/local \
   && curl -L https://dl.bintray.com/boostorg/release/1.66.0/source/boost_${BOOST_VERSION}.tar.gz | tar -xzf - \
   && ln -s ../boost_${BOOST_VERSION}/boost include/boost

# protobuf v3.11.1
ENV PROTOBUF_VERSION=3.11.1
WORKDIR /tmp/unique_for_protobuf
RUN curl -L https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd protobuf-${PROTOBUF_VERSION} && \
    ./configure CXXFLAGS="-fPIC -std=c++11" --prefix=/usr/local && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /tmp/unique_for_protobuf

# Install DALI 
WORKDIR /tmp/unique_for_dali
RUN conda install -y -c conda-forge libtiff>=4.1.0 libjpeg-turbo>=2.0.3
ENV LD_LIBRARY_PATH "/usr/local/lib/:${LD_LIBRARY_PATH}"
RUN git clone --recursive https://github.com/bl0/DALI DALI && \
    cd DALI && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
        -D JPEG_INCLUDE_DIR=/opt/conda/pkgs/libjpeg-turbo-2.0.3-h516909a_1/include \
        -D JPEG_LIBRARY=/opt/conda/pkgs/libjpeg-turbo-2.0.3-h516909a_1/lib/libjpeg.so \
        -D FFMPEG_ROOT_DIR=/usr/local \
        .. && \
    make -j $(nproc) install && \
    pip install dali/python && \
    rm -rf /tmp/unique_for_dali

WORKDIR /root
