# Base CUDAToolkit image to develop DALI, comment/uncomment 11.x or 12.x
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Add test toolchain for later gcc versions
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test

# Add required packages for DALI_deps
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git g++ wget unzip ninja-build autoconf libtool yasm vim \
    unzip python3 python-is-python3

ARG DALI_DEPS_VERSION_SHA
ENV DALI_DEPS_VERSION_SHA=${DALI_DEPS_VERSION_SHA}
RUN git clone https://github.com/NVIDIA/DALI_deps /opt/dali_deps && \
    cd /opt/dali_deps && \
    git checkout ${DALI_DEPS_VERSION_SHA} && \
    git submodule init && \
    git submodule update --depth 1 --init --recursive
RUN cd /opt/dali_deps && /opt/dali_deps/build_scripts/build_deps.sh
RUN rm -r /opt/dali_deps

# Add more packages for DALI
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
