#!/bin/bash -e

export TMP_REPO_DIR=$(mktemp -d)
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
    export DALI_REPO_DIR=$(pwd)
    cp $THIS_PATH/* $TMP_REPO_DIR
    git config --global user.name "Test"
    git config --global user.email "test@test.test"
      pushd $TMP_REPO_DIR
      git init
      git add main_stub.cc CMakeLists.txt
      git commit -m 'initial commit'
        pushd $DALI_REPO_DIR
        git init
        git add cmake dali third_party tools CMakeLists.txt DALI_EXTRA_VERSION VERSION
        git commit -m 'The DALI' -q
        popd
      git submodule add $DALI_REPO_DIR
      git submodule sync --recursive
      git submodule update --init --recursive
      cmake -D CUDA_TARGET_ARCHS=60 -D CMAKE_BUILD_TYPE=Debug .
      make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
      popd
}

pushd ../..
source ./qa/test_template.sh
popd