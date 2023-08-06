# Base CUDAToolkit image to develop DALI, comment/uncomment 11.x or 12.x
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Add required packages for DALI_deps
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git g++ wget unzip ninja-build autoconf libtool yasm vim \
    software-properties-common unzip python3 python-is-python3

# Add nvJPEG2k and nvCOMP since it isn't included in CUDA Toolkit
RUN NVJPEG2K_VERSION=0.7.0.31-1 && \
    NVCOMP_VERSION=2.6.1 && \
    CUDA_VERSION_MAJOR=12 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libnvjpeg2k0=${NVJPEG2K_VERSION} libnvjpeg2k-dev=${NVJPEG2K_VERSION} && \
    cp /usr/include/libnvjpeg2k/${CUDA_VERSION_MAJOR}/* /usr/local/cuda/include/ && \
    cp /usr/lib/x86_64-linux-gnu/libnvjpeg2k/${CUDA_VERSION_MAJOR}/* /usr/local/cuda/lib64/ && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir nvcomp && \
    cd nvcomp && \
    wget https://developer.download.nvidia.com/compute/nvcomp/${NVCOMP_VERSION}/local_installers/nvcomp_${NVCOMP_VERSION}_x86_64_${CUDA_VERSION_MAJOR}.x.tgz  && \
    tar -xvf nvcomp*.tgz && \
    cp -rv include/nvcomp* /usr/local/cuda/include/ && \
    cp -v lib/*.so /usr/local/cuda/lib64/ && \
    cd .. && \
    rm -rf nvcomp

# Clone, Build, Install then remove DALI Deps
ARG DALI_DEPS_VERSION_SHA
ENV DALI_DEPS_VERSION_SHA=${DALI_DEPS_VERSION_SHA}
RUN git clone https://github.com/NVIDIA/DALI_deps /opt/dali_deps && \
    cd /opt/dali_deps && \
    git checkout ${DALI_DEPS_VERSION_SHA} && \
    git submodule init && \
    git submodule update --depth 1 --init --recursive
RUN cd /opt/dali_deps && /opt/dali_deps/build_scripts/build_deps.sh
RUN rm -r /opt/dali_deps

# Add more packages for DALI + Test Toolchain for Newer GCC Versions
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-distutils python3-clang python3-pip pre-commit shellcheck libgl1 \
    git-lfs pkg-config clang clang-format g++-13 gcc-13 libtbb-dev 

RUN useradd -rm -d /home/dev -s /bin/bash -G sudo -U -u 1000 dev && \
    mkdir /opt/dali_extra && chown -R dev /opt/dali_extra

USER dev

# Add DALI_extra, the correct commit will be checked as a part of the
# running tests scripts defined in DALI_EXTRA_VERSION
RUN git clone https://github.com/NVIDIA/DALI_extra.git /opt/dali_extra

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENTRYPOINT /bin/bash
WORKDIR /home/dev
