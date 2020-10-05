#!/bin/bash -e

do_once() {
  apt-get update
  apt-get -y install cmake
}

test_body() {
    export TMP_PATH="$(mktemp -d)"
    pushd $TMP_PATH
    cmake -D CUDA_TARGET_ARCHS=60 .
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
