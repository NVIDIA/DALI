#!/bin/bash -e

export THIS_PATH=`pwd`

do_once() {
  apt-get update && apt-get -y install wget
  wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-Linux-x86_64.sh
  bash cmake-*.sh --skip-license --prefix=/usr
  rm cmake-*.sh
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
