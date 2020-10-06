#!/bin/bash -e

export THIS_PATH=`pwd`

do_once() {
  apt-get update
  apt-get -y install unzip wget build-essential
  CMAKE_VERSION=3.13                                                             && \
  CMAKE_BUILD=3.13.5                                                             && \
  wget -nv https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz && \
  tar -xf cmake-${CMAKE_BUILD}.tar.gz                                            && \
  cd cmake-${CMAKE_BUILD}                                                        && \
  ./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l)                && \
  make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install                      && \
  rm -rf /cmake-${CMAKE_BUILD}
}

test_body() {
    pushd $THIS_PATH
    cmake -D CUDA_TARGET_ARCHS=60 .
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
