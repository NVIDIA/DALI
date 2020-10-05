#!/bin/bash -e

test_body() {
    export TMP_PATH="$(mktemp -d)"
    cd ..
    export DALI_REPO_DIR=$(pwd)
    pushd $TMP_PATH
    git init
    git submodule add $DALI_REPO_DIR
    pushd *
    git submodule sync --recursive
    git submodule update --init --recursive
    popd
    touch main_stub.cc
    mkdir build
    pushd build
    cmake -D CUDA_TARGET_ARCHS=60 -D CMAKE_BUILD_TYPE=Debug ..
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    popd
    popd
}

pushd ../..
source ./qa/test_template.sh
popd
